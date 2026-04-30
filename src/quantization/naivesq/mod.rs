use anyhow::{ensure, Result};

use crate::quantization::VectorQuantizer;

pub struct NaiveSqQuantizer {
    bits: u8,
    dims: usize,
    mins: Vec<f32>,
    spans: Vec<f32>,
    codes: Vec<Vec<u8>>,
}

impl NaiveSqQuantizer {
    pub fn new(dims: usize, bits: u8) -> Self {
        Self {
            bits,
            dims,
            mins: Vec::new(),
            spans: Vec::new(),
            codes: Vec::new(),
        }
    }
}

impl VectorQuantizer for NaiveSqQuantizer {
    fn name(&self) -> &'static str {
        "naivesq"
    }

    fn bits(&self) -> u8 {
        self.bits
    }

    fn encode(&mut self, docs: &[Vec<f32>]) -> Result<()> {
        ensure!((1..=8).contains(&self.bits), "naivesq supports bits 1..=8");
        self.mins = vec![f32::INFINITY; self.dims];
        let mut maxs = vec![f32::NEG_INFINITY; self.dims];
        for doc in docs {
            ensure!(doc.len() == self.dims, "dimension mismatch");
            for dim in 0..self.dims {
                self.mins[dim] = self.mins[dim].min(doc[dim]);
                maxs[dim] = maxs[dim].max(doc[dim]);
            }
        }
        self.spans = self
            .mins
            .iter()
            .zip(maxs.iter())
            .map(|(&lo, &hi)| (hi - lo).max(1e-9))
            .collect();

        let levels = ((1u32 << self.bits) - 1) as f32;
        self.codes = docs
            .iter()
            .map(|doc| {
                (0..self.dims)
                    .map(|dim| {
                        let normalized =
                            ((doc[dim] - self.mins[dim]) / self.spans[dim]).clamp(0.0, 1.0);
                        (normalized * levels).round() as u8
                    })
                    .collect()
            })
            .collect();
        Ok(())
    }

    fn score(&self, query: &[f32], doc_idx: usize) -> f32 {
        let levels = ((1u32 << self.bits) - 1) as f32;
        self.codes[doc_idx]
            .iter()
            .enumerate()
            .map(|(dim, &code)| {
                let value = self.mins[dim] + (code as f32) * (self.spans[dim] / levels);
                value * query[dim]
            })
            .sum()
    }

    fn bytes_per_vector(&self) -> usize {
        (self.dims * self.bits as usize).div_ceil(8)
    }
}
