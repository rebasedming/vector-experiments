use anyhow::{ensure, Result};

use crate::dataset::top_k_by_score;
use crate::quantization::turboquant::{TurboQuantQuery, TurboQuantizer};
use crate::quantization::VectorQuantizer;

pub struct TurboQuantBench {
    inner: TurboQuantizer,
    records: Vec<Vec<u8>>,
}

impl TurboQuantBench {
    pub fn new(dims: usize, bits: u8, seed: u64) -> Self {
        Self {
            inner: TurboQuantizer::new(dims, Some(bits), Some(seed)),
            records: Vec::new(),
        }
    }
}

impl VectorQuantizer for TurboQuantBench {
    fn name(&self) -> &'static str {
        "turboquant"
    }

    fn bits(&self) -> u8 {
        self.inner.bit_width
    }

    fn encode(&mut self, docs: &[Vec<f32>]) -> Result<()> {
        ensure!(!docs.is_empty(), "docs must not be empty");
        self.records = docs.iter().map(|doc| self.inner.encode(doc)).collect();
        Ok(())
    }

    fn score(&self, query: &[f32], doc_idx: usize) -> f32 {
        let query = TurboQuantQuery::new(&self.inner, query);
        query.estimate_ip(&self.records[doc_idx])
    }

    fn top_k(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
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
        self.inner.bytes_per_record()
    }
}
