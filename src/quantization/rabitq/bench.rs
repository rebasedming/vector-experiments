use anyhow::{ensure, Result};

use crate::metrics::top_k_by_score;
use crate::quantization::rabitq::{
    bytes_per_record, prepare_query, quantizer, record, DynamicRotator, Metric, RabitqConfig,
    RotatorType,
};
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
    bits: u8,
    padded_dims: usize,
    rotator: DynamicRotator,
    variant: RabitqVariant,
    config: RabitqConfig,
    records: Vec<Vec<u8>>,
}

impl RabitqBench {
    pub fn new(dims: usize, bits: u8, seed: u64, variant: RabitqVariant) -> Self {
        let rotator = DynamicRotator::new(dims, RotatorType::FhtKacRotator, seed);
        let padded_dims = rotator.padded_dim();
        let config = match variant {
            RabitqVariant::Fixed => RabitqConfig::faster(padded_dims, bits as usize, seed),
            RabitqVariant::Optimal => RabitqConfig::new(bits as usize),
        };
        Self {
            dims,
            bits,
            padded_dims,
            rotator,
            variant,
            config,
            records: Vec::new(),
        }
    }
}

impl VectorQuantizer for RabitqBench {
    fn name(&self) -> &'static str {
        "rabitq"
    }

    fn variant(&self) -> &'static str {
        self.variant.name()
    }

    fn bits(&self) -> u8 {
        self.bits
    }

    fn encode(&mut self, docs: &[Vec<f32>]) -> Result<()> {
        ensure!((1..=16).contains(&self.bits), "rabitq supports bits 1..=16");
        let rotated_zero_centroid = vec![0.0f32; self.padded_dims];
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
        Ok(())
    }

    fn score(&self, query: &[f32], doc_idx: usize) -> f32 {
        let q = prepare_query(
            &self.rotator,
            query,
            (self.bits as usize).saturating_sub(1),
            Metric::InnerProduct,
        );
        -q.estimate_distance_from_record(&self.records[doc_idx], self.padded_dims, 0.0)
    }

    fn top_k(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let q = prepare_query(
            &self.rotator,
            query,
            (self.bits as usize).saturating_sub(1),
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
        bytes_per_record(self.padded_dims, (self.bits as usize).saturating_sub(1))
    }
}
