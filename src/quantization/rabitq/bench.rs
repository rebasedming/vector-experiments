use std::borrow::Cow;

use anyhow::{ensure, Result};

use crate::metrics::top_k_by_score;
use crate::quantization::rabitq::{
    bytes_per_record, prepare_query, quantizer, record, DynamicRotator, Metric, RabitqConfig,
    RotatorType,
};
use crate::quantization::rabitq::transposed::{is_supported_ex_bits, RabitqBatch, BATCH_DOCS};
use crate::quantization::factory::RECALL_QUANT_BITS;
use crate::quantization::VectorQuantizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RabitqVariant {
    Fixed,
    Optimal,
}

impl RabitqVariant {
    pub fn name(self) -> &'static str {
        match self {
            RabitqVariant::Fixed => "fixed",
            RabitqVariant::Optimal => "optimal",
        }
    }
}

pub struct RabitqBench {
    dims: usize,
    padded_dims: usize,
    rotator: DynamicRotator,
    variant: RabitqVariant,
    config: RabitqConfig,
    use_transposed: bool,
    records: Vec<Vec<u8>>,
    batches: Vec<RabitqBatch>,
    doc_count: usize,
}

impl RabitqBench {
    pub fn new(dims: usize, seed: u64, variant: RabitqVariant) -> Self {
        let bits = RECALL_QUANT_BITS as usize;
        let rotator = DynamicRotator::new(dims, RotatorType::FhtKacRotator, seed);
        let padded_dims = rotator.padded_dim();
        let config = match variant {
            RabitqVariant::Fixed => RabitqConfig::faster(padded_dims, bits, seed),
            RabitqVariant::Optimal => RabitqConfig::new(bits),
        };
        Self {
            dims,
            padded_dims,
            rotator,
            variant,
            config,
            use_transposed: false,
            records: Vec::new(),
            batches: Vec::new(),
            doc_count: 0,
        }
    }

    fn ex_bits(&self) -> usize {
        (RECALL_QUANT_BITS as usize).saturating_sub(1)
    }

    fn top_k_transposed(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let q = prepare_query(&self.rotator, query, self.ex_bits(), Metric::InnerProduct);
        let mut scored = Vec::with_capacity(doc_ids.len());

        for (batch_idx, batch) in self.batches.iter().enumerate() {
            let scores = batch.score(&q, self.padded_dims, self.ex_bits());
            let start = batch_idx * BATCH_DOCS;
            let end = (start + BATCH_DOCS).min(doc_ids.len());
            for (slot, &doc_id) in doc_ids[start..end].iter().enumerate() {
                scored.push((doc_id, scores[slot]));
            }
        }

        top_k_by_score(scored, k)
    }
}

impl VectorQuantizer for RabitqBench {
    fn name(&self) -> &'static str {
        "rabitq"
    }

    fn variant(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.variant.name())
    }

    fn scoring_layout(&self) -> &'static str {
        if self.use_transposed {
            "transposed"
        } else {
            "doc-major"
        }
    }

    fn bits(&self) -> u8 {
        RECALL_QUANT_BITS
    }

    fn set_transposed(&mut self, enabled: bool) -> bool {
        self.use_transposed = enabled && is_supported_ex_bits(self.ex_bits());
        self.use_transposed
    }

    fn encode(&mut self, docs: &[Vec<f32>]) -> Result<()> {
        let rotated_zero_centroid = vec![0.0f32; self.padded_dims];
        self.doc_count = docs.len();
        self.records = docs
            .iter()
            .map(|doc| {
                let rotated = self.rotator.rotate(doc);
                let qv = quantizer::quantize_with_centroid(
                    &rotated,
                    &rotated_zero_centroid,
                    &self.config,
                    Metric::InnerProduct,
                );
                record::pack(&qv)
            })
            .collect();
        self.batches.clear();
        if self.use_transposed {
            let ex_bits = self.ex_bits();
            self.batches = self
                .records
                .chunks(BATCH_DOCS)
                .map(|chunk| {
                    let refs: Vec<&[u8]> =
                        chunk.iter().map(|record| record.as_slice()).collect();
                    RabitqBatch::encode(&refs, self.padded_dims, ex_bits)
                })
                .collect();
        }
        Ok(())
    }

    fn score(&self, query: &[f32], doc_idx: usize) -> f32 {
        let q = prepare_query(
            &self.rotator,
            query,
            self.ex_bits(),
            Metric::InnerProduct,
        );
        -q.estimate_distance_from_record(&self.records[doc_idx], self.padded_dims, 0.0)
    }

    fn top_k(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.use_transposed {
            return self.top_k_transposed(doc_ids, query, k);
        }
        let q = prepare_query(
            &self.rotator,
            query,
            self.ex_bits(),
            Metric::InnerProduct,
        );
        top_k_by_score(
            self.records
                .iter()
                .zip(doc_ids.iter().copied())
                .map(|(record, doc_id)| {
                    (
                        doc_id,
                        -q.estimate_distance_from_record(record, self.padded_dims, 0.0),
                    )
                }),
            k,
        )
    }

    fn bytes_per_vector(&self) -> usize {
        if self.use_transposed && self.doc_count > 0 {
            self.total_bytes().div_ceil(self.doc_count)
        } else {
            bytes_per_record(self.padded_dims, self.ex_bits())
        }
    }

    fn total_bytes(&self) -> usize {
        if self.use_transposed {
            self.batches.iter().map(RabitqBatch::bytes).sum()
        } else {
            self.records.iter().map(Vec::len).sum()
        }
    }
}
