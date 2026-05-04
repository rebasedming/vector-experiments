//! Transposed 32-doc batch layout for RaBitQ brute-force scoring.
//!
//! The doc-major RaBitQ record stores one vector contiguously:
//!
//! ```text
//! [binary bits][extended bits][per-vector scalars]
//! ```
//!
//! This layout instead groups 32 records and stores each binary-code byte
//! column-major, so the existing FastScan LUT scorer can process 32 documents
//! at once. For the benchmark's 5-bit setting (`ex_bits = 4`), the extended
//! code bytes are transposed the same way and scored without rebuilding
//! doc-major records.

use super::distance::RaBitQQuery;
use super::fastscan::{self, BATCH_SIZE};
use super::record;

pub const BATCH_DOCS: usize = BATCH_SIZE;

#[inline]
pub fn is_supported_ex_bits(ex_bits: usize) -> bool {
    ex_bits == 4
}

#[inline]
pub fn batch_bytes(padded_dims: usize, ex_bits: usize) -> usize {
    let binary_bytes = padded_dims.div_ceil(8) * BATCH_DOCS;
    let ex_bytes = record::ex_bytes(padded_dims, ex_bits) * BATCH_DOCS;
    let scalar_bytes = 4 * BATCH_DOCS * 4;
    binary_bytes + ex_bytes + scalar_bytes
}

pub struct RabitqBatch {
    binary: Vec<u8>,
    ex: Vec<u8>,
    f_add: [f32; BATCH_DOCS],
    f_rescale: [f32; BATCH_DOCS],
    f_add_ex: [f32; BATCH_DOCS],
    f_rescale_ex: [f32; BATCH_DOCS],
}

impl RabitqBatch {
    pub fn encode(records: &[&[u8]], padded_dims: usize, ex_bits: usize) -> Self {
        assert!(
            records.len() <= BATCH_DOCS,
            "RabitqBatch accepts at most {BATCH_DOCS} records"
        );
        assert!(
            is_supported_ex_bits(ex_bits),
            "transposed RaBitQ supports ex_bits == 4 only (5-bit RaBitQ), got {ex_bits}"
        );

        let binary_bytes = padded_dims.div_ceil(8);
        let ex_bytes = record::ex_bytes(padded_dims, ex_bits);
        let scalar_offset = binary_bytes + ex_bytes;
        let mut binary = vec![0u8; binary_bytes * BATCH_DOCS];
        let mut ex = vec![0u8; ex_bytes * BATCH_DOCS];
        let mut f_add = [0.0f32; BATCH_DOCS];
        let mut f_rescale = [0.0f32; BATCH_DOCS];
        let mut f_add_ex = [0.0f32; BATCH_DOCS];
        let mut f_rescale_ex = [0.0f32; BATCH_DOCS];

        let binary_refs: Vec<&[u8]> = records
            .iter()
            .map(|record| &record[..binary_bytes])
            .collect();
        fastscan::pack_batch_simple(&binary_refs, binary_bytes, &mut binary);

        debug_assert_eq!(ex_bits, 4);
        for col in 0..ex_bytes {
            for (slot, record) in records.iter().enumerate() {
                ex[col * BATCH_DOCS + slot] = record[binary_bytes + col];
            }
        }

        for (slot, record) in records.iter().enumerate() {
            f_add[slot] = read_f32(record, scalar_offset + 8);
            f_rescale[slot] = read_f32(record, scalar_offset + 12);
            f_add_ex[slot] = read_f32(record, scalar_offset + 24);
            f_rescale_ex[slot] = read_f32(record, scalar_offset + 28);
        }

        Self {
            binary,
            ex,
            f_add,
            f_rescale,
            f_add_ex,
            f_rescale_ex,
        }
    }

    pub fn score(&self, query: &RaBitQQuery, padded_dims: usize, ex_bits: usize) -> [f32; BATCH_DOCS] {
        let binary_bytes = padded_dims.div_ceil(8);
        let mut accum = [0u32; BATCH_DOCS];
        let mut binary_dots = [0.0f32; BATCH_DOCS];
        fastscan::accumulate_batch(&self.binary, query.lut().lut_u8(), binary_bytes, &mut accum);
        fastscan::denormalize_batch(
            &accum,
            query.lut().delta(),
            query.lut().sum_vl(),
            &mut binary_dots,
        );

        let mut scores = [0.0f32; BATCH_DOCS];
        debug_assert_eq!(ex_bits, 4);
        let ex_dots = self.ex4_dots(query.rotated_query(), padded_dims);
        let binary_scale = query.binary_scale();
        let kbx_sum_q = query.kbx_sum_q();
        for slot in 0..BATCH_DOCS {
            let term = binary_scale * binary_dots[slot] + ex_dots[slot] + kbx_sum_q;
            scores[slot] = -(self.f_add_ex[slot] + self.f_rescale_ex[slot] * term);
        }
        scores
    }

    #[inline]
    pub fn bytes(&self) -> usize {
        self.binary.len() + self.ex.len() + 4 * BATCH_DOCS * 4
    }

    fn ex4_dots(&self, rotated_query: &[f32], padded_dims: usize) -> [f32; BATCH_DOCS] {
        let mut dots = [0.0f32; BATCH_DOCS];
        let ex_bytes = padded_dims.div_ceil(2);
        for col in 0..ex_bytes {
            let dim = col * 2;
            let q0 = rotated_query[dim];
            let q1 = rotated_query.get(dim + 1).copied().unwrap_or(0.0);
            let column = &self.ex[col * BATCH_DOCS..(col + 1) * BATCH_DOCS];
            for slot in 0..BATCH_DOCS {
                let byte = column[slot];
                dots[slot] += (byte & 0x0F) as f32 * q0 + (byte >> 4) as f32 * q1;
            }
        }
        dots
    }
}

#[inline]
fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use crate::quantization::rabitq::quantizer::{quantize_with_centroid, RabitqConfig};
    use crate::quantization::rabitq::{DynamicRotator, Metric, RotatorType};

    use super::*;

    fn unit(d: usize, seed: u64) -> Vec<f32> {
        use rand::prelude::*;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = Normal::new(0.0_f32, 1.0).unwrap();
        let mut v: Vec<f32> = (0..d).map(|_| n.sample(&mut rng)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn transposed_scores_match_doc_major_ex4() {
        let dims = 128;
        let bits = 5;
        let ex_bits = bits - 1;
        let rotator = DynamicRotator::new(dims, RotatorType::FhtKacRotator, 42);
        let padded_dims = rotator.padded_dim();
        let config = RabitqConfig::faster(padded_dims, bits, 42);
        let zero = vec![0.0f32; padded_dims];
        let docs: Vec<Vec<f32>> = (0..17).map(|idx| unit(dims, 1000 + idx)).collect();
        let records: Vec<Vec<u8>> = docs
            .iter()
            .map(|doc| {
                let rotated = rotator.rotate(doc);
                let qv = quantize_with_centroid(&rotated, &zero, &config, Metric::InnerProduct);
                record::pack(&qv)
            })
            .collect();

        let refs: Vec<&[u8]> = records.iter().map(|record| record.as_slice()).collect();
        let batch = RabitqBatch::encode(&refs, padded_dims, ex_bits);
        let query_vec = unit(dims, 9999);
        let query = RaBitQQuery::new(&query_vec, &rotator, ex_bits, Metric::InnerProduct);
        let scores = batch.score(&query, padded_dims, ex_bits);

        for (idx, record) in records.iter().enumerate() {
            let expected = -query.estimate_distance_from_record(record, padded_dims, 0.0);
            assert!(
                (scores[idx] - expected).abs() < 5e-3,
                "idx={idx} transposed={} doc_major={expected}",
                scores[idx]
            );
        }
    }

    #[test]
    fn transposed_scores_match_doc_major_optimal_ex4() {
        let dims = 128;
        let bits = 5;
        let ex_bits = bits - 1;
        let rotator = DynamicRotator::new(dims, RotatorType::FhtKacRotator, 43);
        let padded_dims = rotator.padded_dim();
        let config = RabitqConfig::new(bits as usize);
        let zero = vec![0.0f32; padded_dims];
        let docs: Vec<Vec<f32>> = (0..12).map(|idx| unit(dims, 2000 + idx)).collect();
        let records: Vec<Vec<u8>> = docs
            .iter()
            .map(|doc| {
                let rotated = rotator.rotate(doc);
                let qv = quantize_with_centroid(&rotated, &zero, &config, Metric::InnerProduct);
                record::pack(&qv)
            })
            .collect();

        let refs: Vec<&[u8]> = records.iter().map(|record| record.as_slice()).collect();
        let batch = RabitqBatch::encode(&refs, padded_dims, ex_bits);
        let query_vec = unit(dims, 7777);
        let query = RaBitQQuery::new(&query_vec, &rotator, ex_bits, Metric::InnerProduct);
        let scores = batch.score(&query, padded_dims, ex_bits);

        for (idx, record) in records.iter().enumerate() {
            let expected = -query.estimate_distance_from_record(record, padded_dims, 0.0);
            assert!(
                (scores[idx] - expected).abs() < 5e-3,
                "optimal idx={idx} transposed={} doc_major={expected}",
                scores[idx]
            );
        }
    }

    #[test]
    fn transposed_partial_batch_matches_doc_major() {
        let dims = 96;
        let bits = 5;
        let ex_bits = bits - 1;
        let rotator = DynamicRotator::new(dims, RotatorType::FhtKacRotator, 44);
        let padded_dims = rotator.padded_dim();
        let config = RabitqConfig::faster(padded_dims, bits, 44);
        let zero = vec![0.0f32; padded_dims];
        let n = 7usize;
        let docs: Vec<Vec<f32>> = (0..n).map(|idx| unit(dims, 3000u64 + idx as u64)).collect();
        let records: Vec<Vec<u8>> = docs
            .iter()
            .map(|doc| {
                let rotated = rotator.rotate(doc);
                let qv = quantize_with_centroid(&rotated, &zero, &config, Metric::InnerProduct);
                record::pack(&qv)
            })
            .collect();

        let refs: Vec<&[u8]> = records.iter().map(|record| record.as_slice()).collect();
        let batch = RabitqBatch::encode(&refs, padded_dims, ex_bits);
        let query_vec = unit(dims, 8888);
        let query = RaBitQQuery::new(&query_vec, &rotator, ex_bits, Metric::InnerProduct);
        let scores = batch.score(&query, padded_dims, ex_bits);

        for idx in 0..n {
            let expected =
                -query.estimate_distance_from_record(&records[idx], padded_dims, 0.0);
            assert!(
                (scores[idx] - expected).abs() < 5e-3,
                "partial idx={idx} transposed={} doc_major={expected}",
                scores[idx]
            );
        }
    }
}
