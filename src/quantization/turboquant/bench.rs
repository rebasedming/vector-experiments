use std::borrow::Cow;

use anyhow::{ensure, Result};

use crate::metrics::top_k_by_score;
use crate::quantization::turboquant::transposed::{
    batch_bytes, encode_batch, is_supported_bit_width, score_batch, BatchedQueryLut, BATCH_DOCS,
};
use crate::quantization::turboquant::{QjlProjectionKind, TurboQuantQuery, TurboQuantizer};
use crate::quantization::VectorQuantizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantVariant {
    Srht,
    GaussianQjl,
}

impl TurboQuantVariant {
    pub fn name(self) -> &'static str {
        match self {
            TurboQuantVariant::Srht => "srht-qjl",
            TurboQuantVariant::GaussianQjl => "gaussian-qjl",
        }
    }
}

pub struct TurboQuantBench {
    inner: TurboQuantizer,
    variant: TurboQuantVariant,
    use_transposed: bool,
    records: Vec<Vec<u8>>,
    batches: Vec<Vec<u8>>,
    doc_count: usize,
}

impl TurboQuantBench {
    pub fn new(dims: usize, seed: u64, variant: TurboQuantVariant) -> Self {
        let qjl_kind = match variant {
            TurboQuantVariant::Srht => QjlProjectionKind::Srht,
            TurboQuantVariant::GaussianQjl => QjlProjectionKind::Gaussian,
        };
        Self {
            inner: TurboQuantizer::new_with_qjl_projection(dims, Some(seed), qjl_kind),
            variant,
            use_transposed: false,
            records: Vec::new(),
            batches: Vec::new(),
            doc_count: 0,
        }
    }

    fn top_k_transposed(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let query = TurboQuantQuery::new(&self.inner, query);
        let lut = BatchedQueryLut::new(&query);
        let mut scored = Vec::with_capacity(doc_ids.len());
        let mut out = [0.0f32; BATCH_DOCS];

        for (batch_idx, batch) in self.batches.iter().enumerate() {
            score_batch(&lut, batch, &mut out);
            let start = batch_idx * BATCH_DOCS;
            let end = (start + BATCH_DOCS).min(doc_ids.len());
            for (slot, &doc_id) in doc_ids[start..end].iter().enumerate() {
                scored.push((doc_id, out[slot]));
            }
        }

        top_k_by_score(scored, k)
    }
}

impl VectorQuantizer for TurboQuantBench {
    fn name(&self) -> &'static str {
        "turboquant"
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
        self.inner.bit_width
    }

    fn set_transposed(&mut self, enabled: bool) -> bool {
        self.use_transposed = enabled && is_supported_bit_width(self.inner.bit_width);
        self.use_transposed
    }

    fn encode(&mut self, docs: &[Vec<f32>]) -> Result<()> {
        ensure!(!docs.is_empty(), "docs must not be empty");
        self.doc_count = docs.len();
        self.records = self.inner.encode_many(docs);
        self.batches.clear();
        if self.use_transposed {
            let padded = self.inner.padded_dim;
            let bw = self.inner.bit_width;
            self.batches = self
                .records
                .chunks(BATCH_DOCS)
                .map(|chunk| {
                    let mut batch = vec![0u8; batch_bytes(padded)];
                    let refs: Vec<&[u8]> =
                        chunk.iter().map(|record| record.as_slice()).collect();
                    encode_batch(&refs, padded, bw, &mut batch);
                    batch
                })
                .collect();
        }
        Ok(())
    }

    fn score(&self, query: &[f32], doc_idx: usize) -> f32 {
        let query = TurboQuantQuery::new(&self.inner, query);
        query.estimate_ip(&self.records[doc_idx])
    }

    fn top_k(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.use_transposed {
            return self.top_k_transposed(doc_ids, query, k);
        }
        let query = TurboQuantQuery::new(&self.inner, query);
        top_k_by_score(
            self.records
                .iter()
                .zip(doc_ids.iter().copied())
                .map(|(record, doc_id)| (doc_id, query.estimate_ip(record))),
            k,
        )
    }

    fn bytes_per_vector(&self) -> usize {
        if self.use_transposed && self.doc_count > 0 {
            self.total_bytes().div_ceil(self.doc_count)
        } else {
            self.inner.bytes_per_record()
        }
    }

    fn total_bytes(&self) -> usize {
        if self.use_transposed {
            self.batches.iter().map(Vec::len).sum()
        } else {
            self.records.iter().map(Vec::len).sum()
        }
    }
}
