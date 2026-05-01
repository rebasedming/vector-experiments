use anyhow::{ensure, Result};

use crate::metrics::top_k_by_score;
use crate::quantization::turboquant::bitpack;
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
        let bytes_per_vector = self.bytes_per_vector();
        self.codes = docs
            .iter()
            .map(|doc| {
                let codes: Vec<u8> = (0..self.dims)
                    .map(|dim| {
                        let normalized =
                            ((doc[dim] - self.mins[dim]) / self.spans[dim]).clamp(0.0, 1.0);
                        (normalized * levels).round() as u8
                    })
                    .collect();
                let mut packed = vec![0u8; bytes_per_vector];
                bitpack::pack_into(&codes, self.bits, &mut packed);
                packed
            })
            .collect();
        Ok(())
    }

    fn score(&self, query: &[f32], doc_idx: usize) -> f32 {
        NaiveSqQuery::new(self, query).score_record(&self.codes[doc_idx])
    }

    fn top_k(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let query = NaiveSqQuery::new(self, query);
        top_k_by_score(
            self.codes
                .iter()
                .zip(doc_ids.iter().copied())
                .map(|(record, doc_id)| (doc_id, query.score_record(record))),
            k,
        )
    }

    fn bytes_per_vector(&self) -> usize {
        bitpack::packed_byte_size(self.dims, self.bits)
    }
}

struct NaiveSqQuery {
    bits: u8,
    dims: usize,
    levels: usize,
    lut: Vec<f32>,
}

impl NaiveSqQuery {
    fn new(quantizer: &NaiveSqQuantizer, query: &[f32]) -> Self {
        debug_assert_eq!(query.len(), quantizer.dims);
        let levels = (1usize << quantizer.bits) - 1;
        let mut lut = vec![0.0f32; quantizer.dims * (levels + 1)];
        for dim in 0..quantizer.dims {
            let base = dim * (levels + 1);
            let scale = quantizer.spans[dim] / levels as f32;
            for code in 0..=levels {
                let value = quantizer.mins[dim] + code as f32 * scale;
                lut[base + code] = value * query[dim];
            }
        }
        Self {
            bits: quantizer.bits,
            dims: quantizer.dims,
            levels,
            lut,
        }
    }

    fn score_record(&self, record: &[u8]) -> f32 {
        if self.bits == 5 {
            return self.score_record_b5(record);
        }
        (0..self.dims)
            .map(|dim| {
                let code = unpack_code(record, dim, self.bits) as usize;
                self.lut[dim * (self.levels + 1) + code]
            })
            .sum()
    }

    fn score_record_b5(&self, record: &[u8]) -> f32 {
        debug_assert_eq!(self.bits, 5);
        let levels = self.levels + 1;
        let mut sum = 0.0f32;
        let full_groups = self.dims / 8;
        for group in 0..full_groups {
            let byte = group * 5;
            let dim = group * 8;
            let b0 = record[byte] as u64;
            let b1 = record[byte + 1] as u64;
            let b2 = record[byte + 2] as u64;
            let b3 = record[byte + 3] as u64;
            let b4 = record[byte + 4] as u64;
            let packed = (b0 << 32) | (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;

            let c0 = ((packed >> 35) & 0x1F) as usize;
            let c1 = ((packed >> 30) & 0x1F) as usize;
            let c2 = ((packed >> 25) & 0x1F) as usize;
            let c3 = ((packed >> 20) & 0x1F) as usize;
            let c4 = ((packed >> 15) & 0x1F) as usize;
            let c5 = ((packed >> 10) & 0x1F) as usize;
            let c6 = ((packed >> 5) & 0x1F) as usize;
            let c7 = (packed & 0x1F) as usize;

            let base = dim * levels;
            sum += self.lut[base + c0];
            sum += self.lut[base + levels + c1];
            sum += self.lut[base + 2 * levels + c2];
            sum += self.lut[base + 3 * levels + c3];
            sum += self.lut[base + 4 * levels + c4];
            sum += self.lut[base + 5 * levels + c5];
            sum += self.lut[base + 6 * levels + c6];
            sum += self.lut[base + 7 * levels + c7];
        }

        for dim in full_groups * 8..self.dims {
            let code = unpack_code(record, dim, self.bits) as usize;
            sum += self.lut[dim * levels + code];
        }
        sum
    }
}

fn unpack_code(record: &[u8], dim: usize, bits: u8) -> u8 {
    let bit_offset = dim * bits as usize;
    let byte_idx = bit_offset / 8;
    let bit_idx = bit_offset % 8;
    let hi = record[byte_idx] as u16;
    let lo = if byte_idx + 1 < record.len() {
        record[byte_idx + 1] as u16
    } else {
        0
    };
    let combined = (hi << 8) | lo;
    let shift = 16 - bits as u32 - bit_idx as u32;
    let mask = (1u16 << bits as u32) - 1;
    ((combined >> shift) & mask) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b5_query_matches_generic_score() {
        let docs: Vec<Vec<f32>> = (0..17)
            .map(|doc| {
                (0..13)
                    .map(|dim| ((doc * 11 + dim * 7) % 31) as f32 / 31.0)
                    .collect()
            })
            .collect();
        let query: Vec<f32> = (0..13).map(|dim| dim as f32 / 13.0 - 0.5).collect();
        let mut quantizer = NaiveSqQuantizer::new(13, 5);
        quantizer.encode(&docs).unwrap();
        let q = NaiveSqQuery::new(&quantizer, &query);
        for doc_idx in 0..docs.len() {
            let optimized = q.score_record(&quantizer.codes[doc_idx]);
            let generic: f32 = (0..quantizer.dims)
                .map(|dim| {
                    let code = unpack_code(&quantizer.codes[doc_idx], dim, quantizer.bits);
                    q.lut[dim * (q.levels + 1) + code as usize]
                })
                .sum();
            assert!(
                (optimized - generic).abs() < 1e-5,
                "doc={doc_idx} optimized={optimized} generic={generic}"
            );
        }
    }
}
