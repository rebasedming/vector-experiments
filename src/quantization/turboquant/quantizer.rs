//! TurboQuant encoder.
//!
//! Two-stage data-oblivious vector quantization. Encodes one vector at
//! a time into a fixed-size record; needs no centroid, no training, no
//! per-dataset state beyond a shared rotator + codebook.
//!
//! ## Stages
//!
//! **Stage 1 — MSE.** Rotate the vector with a Haar-like random
//! rotation (we reuse `DynamicRotator::FhtKacRotator` for O(d log d)
//! speed), then scalar-quantize each coordinate with a Lloyd-Max
//! codebook tuned for the coordinate's Beta marginal. Uses `bit_width
//! - 1` bits per coordinate when `bit_width > 1`, else Stage 1 is
//! skipped.
//!
//! **Stage 2 — QJL residual.** Compute the residual `r = rotated -
//! dequant(stage1)`, project it, and store the sign bits together with
//! the residual norm γ. The default projection is a fast SRHT-style
//! FhtKacRotator; `QjlProjectionKind::Gaussian` uses the dense Gaussian
//! projection from the TurboQuant paper.
//!
//! ## Input contract
//!
//! The codebook is tuned for **unit-norm** input — callers are
//! responsible for normalising vectors before `encode` when that
//! matches their metric (e.g. cosine). Non-unit inputs still encode
//! cleanly but waste bits; scalar indices saturate at the extreme
//! centroids and Stage 2 compensates with a larger γ.

use std::sync::Arc;

use super::bitpack;
use super::codebook::{get_or_generate_cached, Codebook};
use super::record::{bytes_per_record, norm_offset, stage2_offset, write_norm};
use crate::quantization::rotation::{DynamicRotator, RotatorType};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// A configured TurboQuant encoder/decoder.
///
/// Cheap to clone (all state is behind `Arc`).
#[derive(Clone)]
pub struct TurboQuantizer {
    /// Input dimensionality.
    pub dim: usize,
    /// Rotator-padded dimensionality (always `>= dim`).
    pub padded_dim: usize,
    /// Total bits per coordinate: Stage 1 uses `bit_width - 1`, Stage 2 uses 1.
    pub bit_width: u8,
    /// Stage 1 Haar-like rotation (applied to the raw input vector).
    rotator: Arc<DynamicRotator>,
    /// Stage 2 QJL projection (applied to the Stage 1 residual).
    qjl_projection: Arc<QjlProjection>,
    /// Scalar codebook for Stage 1.
    codebook: Arc<Codebook>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QjlProjectionKind {
    Srht,
    Gaussian,
}

#[derive(Clone)]
enum QjlProjection {
    Srht(DynamicRotator),
    Gaussian(GaussianProjection),
}

#[derive(Clone)]
struct GaussianProjection {
    dim: usize,
    rows: Vec<f32>,
}

impl GaussianProjection {
    fn new(dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0f32).expect("valid normal distribution");
        let rows = (0..dim * dim).map(|_| normal.sample(&mut rng)).collect();
        Self { dim, rows }
    }

    fn project(&self, vector: &[f32]) -> Vec<f32> {
        debug_assert_eq!(vector.len(), self.dim);
        let mut out = vec![0.0f32; self.dim];
        for row in 0..self.dim {
            let row_values = &self.rows[row * self.dim..(row + 1) * self.dim];
            out[row] = row_values
                .iter()
                .zip(vector.iter())
                .map(|(&a, &b)| a * b)
                .sum();
        }
        out
    }
}

impl QjlProjection {
    fn new(kind: QjlProjectionKind, dim: usize, seed: u64) -> Self {
        match kind {
            QjlProjectionKind::Srht => {
                Self::Srht(DynamicRotator::new(dim, RotatorType::FhtKacRotator, seed))
            }
            QjlProjectionKind::Gaussian => Self::Gaussian(GaussianProjection::new(dim, seed)),
        }
    }

    fn project(&self, vector: &[f32]) -> Vec<f32> {
        match self {
            QjlProjection::Srht(rotator) => rotator.rotate(vector),
            QjlProjection::Gaussian(projection) => projection.project(vector),
        }
    }

    fn estimator_scale(&self, dim: usize) -> f32 {
        let qjl = (std::f32::consts::PI / 2.0).sqrt();
        match self {
            QjlProjection::Srht(_) => qjl / (dim as f32).sqrt(),
            QjlProjection::Gaussian(_) => qjl / dim as f32,
        }
    }
}

/// Total bits per coordinate: Stage 1 uses **4 bits**, Stage 2 uses **1 bit**.
pub const BIT_WIDTH: u8 = 5;

/// Default rotator seed when callers don't specify. The Stage 2 QJL
/// rotation derives its own seed from this via `wrapping_add(GOLDEN)`.
pub const DEFAULT_ROTATOR_SEED: u64 = 42;

impl TurboQuantizer {
    /// Build a quantizer for `dim`-dimensional vectors at fixed [`BIT_WIDTH`].
    pub fn new(dim: usize, rotator_seed: Option<u64>) -> Self {
        Self::new_with_qjl_projection(dim, rotator_seed, QjlProjectionKind::Srht)
    }

    pub fn new_with_qjl_projection(
        dim: usize,
        rotator_seed: Option<u64>,
        qjl_kind: QjlProjectionKind,
    ) -> Self {
        let rotator_seed = rotator_seed.unwrap_or(DEFAULT_ROTATOR_SEED);
        assert!(dim > 0, "dim must be > 0");

        let rotator = DynamicRotator::new(dim, RotatorType::FhtKacRotator, rotator_seed);
        let padded_dim = rotator.padded_dim();

        let qjl_seed = rotator_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let qjl_projection = QjlProjection::new(qjl_kind, padded_dim, qjl_seed);

        let bit_width = BIT_WIDTH;
        let s1_bits = bit_width - 1;
        let codebook = get_or_generate_cached(padded_dim, s1_bits);

        Self {
            dim,
            padded_dim,
            bit_width,
            rotator: Arc::new(rotator),
            qjl_projection: Arc::new(qjl_projection),
            codebook: Arc::new(codebook),
        }
    }

    /// Bytes per encoded record.
    #[inline]
    pub fn bytes_per_record(&self) -> usize {
        bytes_per_record(self.padded_dim, self.bit_width)
    }

    pub(crate) fn rotator(&self) -> &DynamicRotator {
        &self.rotator
    }
    pub(crate) fn codebook(&self) -> &Codebook {
        &self.codebook
    }
    pub(crate) fn qjl_project(&self, vector: &[f32]) -> Vec<f32> {
        self.qjl_projection.project(vector)
    }
    pub(crate) fn qjl_estimator_scale(&self) -> f32 {
        self.qjl_projection.estimator_scale(self.padded_dim)
    }

    /// Encode `vec` into a fresh record byte vector.
    ///
    /// Panics if `vec.len() != self.dim`.
    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.bytes_per_record()];
        self.encode_into(vec, &mut out);
        out
    }

    pub fn encode_many(&self, docs: &[Vec<f32>]) -> Vec<Vec<u8>> {
        match self.qjl_projection.as_ref() {
            QjlProjection::Srht(_) => docs.iter().map(|doc| self.encode(doc)).collect(),
            QjlProjection::Gaussian(projection) => self.encode_many_gaussian(docs, projection),
        }
    }

    /// Reconstruct an approximate rotated-space vector from `record`.
    ///
    /// Writes `padded_dim` floats to `out`. Uses Stage 1 only (the
    /// scalar dequantized indices); Stage 2 sign bits cannot be
    /// reconstructed per-coordinate. The result lives in the rotated
    /// space (apply `rotator().inverse_rotate` to get back to input
    /// space — usually unnecessary, since callers that want to compare
    /// against centroids can rotate the centroids once instead).
    ///
    /// For `BIT_WIDTH`, Stage 1 reconstructs scalar indices only.
    pub fn dequantize_into(&self, record: &[u8], out: &mut [f32]) {
        assert!(
            out.len() >= self.padded_dim,
            "dequantize_into: out buffer too small ({} < {})",
            out.len(),
            self.padded_dim
        );
        for x in out[..self.padded_dim].iter_mut() {
            *x = 0.0;
        }
        use super::bitpack;
        use super::record::stage1_bytes;
        let s1_bits = self.bit_width - 1;
        let s1 = stage1_bytes(record, self.padded_dim, self.bit_width);
        let mut indices = vec![0u8; self.padded_dim];
        bitpack::unpack_into(s1, self.padded_dim, s1_bits, &mut indices);
        for i in 0..self.padded_dim {
            out[i] = self.codebook.dequantize_scalar(indices[i]);
        }
    }

    /// Encode `vec` into `out` (which must be at least `bytes_per_record()`
    /// bytes). Reuses the caller's buffer to avoid a per-doc allocation
    /// on the hot path.
    pub fn encode_into(&self, vec: &[f32], out: &mut [u8]) {
        assert_eq!(vec.len(), self.dim, "TurboQuant encode: dim mismatch");
        assert!(out.len() >= self.bytes_per_record());

        let mut residual = vec![0.0f32; self.padded_dim];
        self.encode_stage1_and_residual(vec, out, &mut residual);

        // Stage 2 — project residual, pack sign bits
        let sr = self.qjl_project(&residual);
        let s1_end = stage2_offset(self.padded_dim, self.bit_width);
        let norm_off = norm_offset(self.padded_dim, self.bit_width);
        pack_projected_signs_into(&sr, &mut out[s1_end..norm_off]);
    }

    fn encode_stage1_and_residual(&self, vec: &[f32], out: &mut [u8], residual: &mut [f32]) {
        assert_eq!(vec.len(), self.dim, "TurboQuant encode: dim mismatch");
        assert!(out.len() >= self.bytes_per_record());
        assert!(residual.len() >= self.padded_dim);

        // Stage 1 — rotate, scalar-quantize, reconstruct in rotated space
        let z = self.rotator.rotate(vec);
        let mut s1_indices = vec![0u8; self.padded_dim];
        let mut gamma_sq = 0.0f64;

        for i in 0..self.padded_dim {
            let idx = self.codebook.quantize_scalar(z[i]);
            s1_indices[i] = idx;
            let r = z[i] - self.codebook.dequantize_scalar(idx);
            residual[i] = r;
            gamma_sq += (r as f64) * (r as f64);
        }
        let gamma = gamma_sq.sqrt() as f32;

        // Zero the packed region (indices / signs) before OR-style packing.
        let norm_off = norm_offset(self.padded_dim, self.bit_width);
        for b in &mut out[..norm_off] {
            *b = 0;
        }

        // Stage 1 packed indices
        let s1_end = stage2_offset(self.padded_dim, self.bit_width);
        let s1_bits = self.bit_width - 1;
        bitpack::pack_into(&s1_indices, s1_bits, &mut out[..s1_end]);
        write_norm(out, self.padded_dim, self.bit_width, gamma);
    }

    fn encode_many_gaussian(
        &self,
        docs: &[Vec<f32>],
        projection: &GaussianProjection,
    ) -> Vec<Vec<u8>> {
        let record_bytes = self.bytes_per_record();
        let mut records = Vec::with_capacity(docs.len());
        let chunk_size = 1024usize;
        let d = self.padded_dim;
        let s1_end = stage2_offset(d, self.bit_width);
        let norm_off = norm_offset(d, self.bit_width);

        for chunk in docs.chunks(chunk_size) {
            let rows = chunk.len();
            let mut residuals = vec![0.0f32; rows * d];
            let mut projected = vec![0.0f32; rows * d];
            let first_record = records.len();

            for (row, doc) in chunk.iter().enumerate() {
                let mut record = vec![0u8; record_bytes];
                let residual = &mut residuals[row * d..(row + 1) * d];
                self.encode_stage1_and_residual(doc, &mut record, residual);
                records.push(record);
            }

            unsafe {
                matrixmultiply::sgemm(
                    rows,
                    d,
                    d,
                    1.0,
                    residuals.as_ptr(),
                    d as isize,
                    1,
                    projection.rows.as_ptr(),
                    1,
                    d as isize,
                    0.0,
                    projected.as_mut_ptr(),
                    d as isize,
                    1,
                );
            }

            for row in 0..rows {
                let signs = &projected[row * d..(row + 1) * d];
                let record = &mut records[first_record + row];
                pack_projected_signs_into(signs, &mut record[s1_end..norm_off]);
            }
        }

        records
    }
}

fn pack_projected_signs_into(projected: &[f32], out: &mut [u8]) {
    let need = projected.len().div_ceil(8);
    debug_assert!(out.len() >= need);
    for byte in &mut out[..need] {
        *byte = 0;
    }
    for (idx, &value) in projected.iter().enumerate() {
        if value >= 0.0 {
            out[idx / 8] |= 1 << (7 - (idx % 8));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_rand(d: usize, seed: u64) -> Vec<f32> {
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
    fn encode_record_size_matches_layout() {
        let tq = TurboQuantizer::new(768, Some(42));
        let v = unit_rand(768, 1);
        let rec = tq.encode(&v);
        assert_eq!(rec.len(), tq.bytes_per_record());
    }

    #[test]
    fn encode_is_deterministic_for_same_seed() {
        let tq_a = TurboQuantizer::new(256, Some(42));
        let tq_b = TurboQuantizer::new(256, Some(42));
        let v = unit_rand(256, 7);
        assert_eq!(tq_a.encode(&v), tq_b.encode(&v));
    }

    #[test]
    fn different_seeds_give_different_records() {
        let tq_a = TurboQuantizer::new(256, Some(42));
        let tq_b = TurboQuantizer::new(256, Some(43));
        let v = unit_rand(256, 7);
        assert_ne!(tq_a.encode(&v), tq_b.encode(&v));
    }

    #[test]
    fn encode_into_matches_encode() {
        let tq = TurboQuantizer::new(256, Some(42));
        let v = unit_rand(256, 11);
        let out1 = tq.encode(&v);
        let mut out2 = vec![0u8; tq.bytes_per_record()];
        tq.encode_into(&v, &mut out2);
        assert_eq!(out1, out2);
    }

    #[test]
    fn gaussian_encode_many_matches_single_encode() {
        let tq =
            TurboQuantizer::new_with_qjl_projection(128, Some(42), QjlProjectionKind::Gaussian);
        let docs: Vec<Vec<f32>> = (0..17).map(|idx| unit_rand(128, idx + 100)).collect();
        let one_by_one: Vec<_> = docs.iter().map(|doc| tq.encode(doc)).collect();
        let batched = tq.encode_many(&docs);
        assert_eq!(one_by_one, batched);
    }
}
