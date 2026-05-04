//! Query-time inner-product estimation against TurboQuant records.
//!
//! A `TurboQuantQuery` is built once per search, then hot-looped to
//! estimate `<query, record>` for every candidate. Keeping the rotated
//! query + QJL-projected query cached avoids redoing O(d log d) work on
//! every candidate.
//!
//! ## IP formula
//!
//! Given encode(`x`) = (stage1_indices `z̃`, stage2_signs `s`, γ), query
//! `y`, Stage 1 rotator `R_1` and Stage 2 QJL projection `R_2`:
//!
//!     ỹ  = R_1 · y                  (`rotated_query`)
//!     z̃_f = dequant(stage1_indices)  (reconstructed rotated vec)
//!     ŷ  = R_2 · ỹ                  (`qjl_query`)
//!
//! Then
//!
//!     <x, y> ≈ <z̃_f, ỹ> + (√(π/2) · γ / √d) · <sign_to_±1(s), ŷ>
//!
//! The first term is Stage 1 MSE; the second is the QJL residual
//! correction (Stage 2). Returning IP — *higher is more similar* —
//! matches what `sort_by_vector_distance.rs` expects after our
//! InnerProduct sign fix, so downstream code stays unchanged.
//!
//! ## NEON SIMD
//!
//! On `aarch64`, the hot path uses NEON kernels for fixed [`super::quantizer::BIT_WIDTH`] (= 5).

use super::quantizer::TurboQuantizer;
use super::record::{read_norm, stage1_bytes, stage2_bytes};

/// Per-query state shared across all candidate evaluations in one search.
pub struct TurboQuantQuery {
    rotated_query: Vec<f32>,
    qjl_query: Vec<f32>,
    /// `sqrt(π/2) / sqrt(padded_dim)` — scale factor on the Stage 2 contribution.
    qjl_scale: f32,
    /// Padded dim (= codebook + record layout dim).
    padded_dim: usize,
    bit_width: u8,
    /// Per-coordinate × per-codebook-entry LUT: `lut[i * K + v] =
    /// rotated_query[i] * codebook.dequantize_scalar(v)` where
    /// K = 2^(bit_width - 1). Used by the scalar fallback. Empty when
    /// the NEON fast path is selected, or when `bit_width == 1`.
    s1_lut: Vec<f32>,
    /// 2^(bit_width - 1) — number of Stage 1 codebook entries; cached so
    /// the hot loop can use a constant when `bit_width` is known.
    s1_levels: usize,
    /// Bare Stage 1 codebook centroids (only populated when the NEON
    /// fast path is selected). Indexed by quantization index.
    codebook: Vec<f32>,
    /// True when the SIMD Stage 1 / Stage 2 kernels are used.
    use_simd: bool,
}

impl TurboQuantQuery {
    /// Prepare `query` for many estimate_ip calls.
    pub fn new(quantizer: &TurboQuantizer, query: &[f32]) -> Self {
        assert_eq!(query.len(), quantizer.dim, "TurboQuantQuery: dim mismatch");
        let rotated = quantizer.rotator().rotate(query);
        let qjl_query = quantizer.qjl_project(&rotated);
        let qjl_scale = quantizer.qjl_estimator_scale();

        let bw = quantizer.bit_width;
        let s1_levels = if bw > 1 { 1usize << (bw - 1) } else { 0 };
        let codebook_ref = quantizer.codebook();

        // SIMD selection: we have NEON Stage 1 kernels for b=4 and b=5,
        // and a NEON Stage 2 kernel for any bit width with d % 8 == 0.
        // We enable SIMD whenever Stage 2 (which always runs) can be
        // vectorized. Stage 1 takes the SIMD path only when both
        // conditions hold; otherwise it falls back to the scalar LUT.
        let stage2_simd_ok = cfg!(target_arch = "aarch64") && quantizer.padded_dim % 8 == 0;
        let stage1_simd_ok = stage2_simd_ok && bw == 5;
        let use_simd = stage2_simd_ok;

        let codebook: Vec<f32> = if stage1_simd_ok {
            (0..s1_levels)
                .map(|v| codebook_ref.dequantize_scalar(v as u8))
                .collect()
        } else {
            Vec::new()
        };

        // s1_lut is always populated when there is a Stage 1 (b > 1).
        // The SIMD kernel uses the bare codebook + rotated_query
        // instead, but the scalar fallback / debug paths still want
        // the LUT, and building it is a one-time per-query cost
        // (≈ d * K * 4 bytes ≈ 24 KiB at d=768, K=8 — negligible
        // against the work of querying 60 K candidates).
        let s1_lut: Vec<f32> = if s1_levels == 0 {
            Vec::new()
        } else {
            let mut lut = Vec::with_capacity(quantizer.padded_dim * s1_levels);
            for i in 0..quantizer.padded_dim {
                let qi = rotated[i];
                for v in 0..s1_levels {
                    lut.push(qi * codebook_ref.dequantize_scalar(v as u8));
                }
            }
            lut
        };

        Self {
            rotated_query: rotated,
            qjl_query,
            qjl_scale,
            padded_dim: quantizer.padded_dim,
            bit_width: bw,
            s1_lut,
            s1_levels,
            codebook,
            use_simd,
        }
    }

    /// Estimate the inner product `<x, query>` for the record encoded from
    /// some full-precision `x`. Higher is more similar.
    #[inline]
    pub fn estimate_ip(&self, record: &[u8]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            if self.use_simd {
                // SAFETY: `use_simd` is set only when the platform is
                // aarch64 and `padded_dim % 8 == 0`. NEON is part of
                // the aarch64 baseline.
                return unsafe { self.estimate_ip_neon(record) };
            }
        }
        self.estimate_ip_scalar(record)
    }

    /// Scalar reference implementation — used as a fallback and for
    /// correctness comparisons against the SIMD kernels.
    #[inline]
    fn estimate_ip_scalar(&self, record: &[u8]) -> f32 {
        let d = self.padded_dim;
        let bw = self.bit_width;

        // Stage 1: reconstruct dequantized rotated vector, dot with
        // rotated query. Skip entirely if bit_width == 1.
        let mut ip = 0.0f32;
        if bw > 1 {
            let s1_bits = (bw - 1) as u32;
            let mask = ((1u16 << s1_bits) - 1) as u16;
            let s1 = stage1_bytes(record, d, bw);
            let levels = self.s1_levels;
            let lut = self.s1_lut.as_slice();
            for i in 0..d {
                let bit_offset = i * s1_bits as usize;
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                let hi: u16 = unsafe { *s1.get_unchecked(byte_idx) }.into();
                let lo: u16 = if byte_idx + 1 < s1.len() {
                    unsafe { *s1.get_unchecked(byte_idx + 1) }.into()
                } else {
                    0
                };
                let combined = (hi << 8) | lo;
                let shift = 16 - s1_bits - bit_idx as u32;
                let idx = ((combined >> shift) & mask) as usize;
                ip += unsafe { *lut.get_unchecked(i * levels + idx) };
            }
        }

        // Stage 2: sign-weighted dot of QJL-projected query, scaled by γ·√(π/2)/√d.
        let s2 = stage2_bytes(record, d, bw);
        let gamma = read_norm(record, d, bw);
        let mut stage2 = 0.0f32;
        let qjl = self.qjl_query.as_slice();
        let full_bytes = d / 8;
        for byte_idx in 0..full_bytes {
            let bits = unsafe { *s2.get_unchecked(byte_idx) };
            let base = byte_idx * 8;
            for j in 0..8 {
                let q = unsafe { *qjl.get_unchecked(base + j) };
                let bit = (bits >> (7 - j)) & 1;
                if bit == 1 {
                    stage2 += q;
                } else {
                    stage2 -= q;
                }
            }
        }
        let tail_bits = d - full_bytes * 8;
        if tail_bits > 0 {
            let bits = s2[full_bytes];
            let base = full_bytes * 8;
            for j in 0..tail_bits {
                let q = self.qjl_query[base + j];
                let bit = (bits >> (7 - j)) & 1;
                if bit == 1 {
                    stage2 += q;
                } else {
                    stage2 -= q;
                }
            }
        }

        ip + gamma * self.qjl_scale * stage2
    }

    /// NEON-vectorized estimator. Delegates to specialized kernels.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn estimate_ip_neon(&self, record: &[u8]) -> f32 {
        let d = self.padded_dim;
        let bw = self.bit_width;
        let s2 = stage2_bytes(record, d, bw);
        let gamma = read_norm(record, d, bw);
        let full_bytes = d / 8;

        let stage2 = neon::stage2_dot_neon(s2, &self.qjl_query, full_bytes);

        let s1 = stage1_bytes(record, d, bw);
        let cb: &[f32; 16] = self.codebook.as_slice().try_into().unwrap_unchecked();
        let ip = neon::stage1_dot_b5_neon(s1, &self.rotated_query, cb, d);

        ip + gamma * self.qjl_scale * stage2
    }

    /// Estimate the squared L2 distance `‖x - query‖²`. Assumes inputs
    /// are unit-norm (so `‖x - y‖² = 2 - 2·<x, y>`); used by
    /// cosine-ordering callers.
    #[inline]
    pub fn estimate_l2sq_unit(&self, record: &[u8]) -> f32 {
        2.0 - 2.0 * self.estimate_ip(record)
    }

    /// `(s1_lut, K)` view: `s1_lut[i*K + v] = rotated_query[i] *
    /// codebook[v]`. Exposed for the batched scorer in
    /// `transposed.rs` so it can reuse the cached per-query tables.
    pub fn s1_lut_view(&self) -> (&[f32], usize) {
        (self.s1_lut.as_slice(), self.s1_levels)
    }

    /// QJL-projected query, length `padded_dim`. Stage-2 scoring
    /// dots this against per-doc sign bits.
    pub fn qjl_query_view(&self) -> &[f32] {
        &self.qjl_query
    }

    /// Scalar `√(π/2) / √padded_dim` Stage-2 multiplier.
    pub fn qjl_scale_value(&self) -> f32 {
        self.qjl_scale
    }

    /// Padded vector dimension (`= TurboQuantizer::padded_dim`).
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Bit width (`TurboQuantizer::BIT_WIDTH`).
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }
}

/// Precomputed XOR sign masks: `SIGN_MASK_TABLE[byte][j]` is `0` if bit
/// `j` (MSB-first) of `byte` is 1, else `0x8000_0000`. XORing a float's
/// bit pattern with this mask flips its sign exactly when the
/// corresponding bit is 0 — i.e. `f → +f if bit==1 else -f`. This makes
/// the Stage 2 inner loop branchless.
///
/// 256 × 8 × 4 B = 8 KiB, fits comfortably in L1d.
const SIGN_MASK_TABLE: [[u32; 8]; 256] = build_sign_mask_table();

const fn build_sign_mask_table() -> [[u32; 8]; 256] {
    let mut t = [[0u32; 8]; 256];
    let mut b: usize = 0;
    while b < 256 {
        let mut j = 0;
        while j < 8 {
            let bit = (b >> (7 - j)) & 1;
            t[b][j] = if bit == 1 { 0 } else { 0x8000_0000 };
            j += 1;
        }
        b += 1;
    }
    t
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    use super::SIGN_MASK_TABLE;

    /// Stage 2 inner loop: `Σ_byte Σ_j (bit_j ? +qjl[base+j] : -qjl[base+j])`.
    ///
    /// Per byte we load 8 floats from `qjl`, look up an 8-lane XOR sign
    /// mask from the byte, XOR, and accumulate. We process 2 bytes (16
    /// floats) per iteration with 4 independent accumulators to hide
    /// FMA/ALU latency on Apple-silicon NEON.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn stage2_dot_neon(s2: &[u8], qjl: &[f32], full_bytes: usize) -> f32 {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let qptr = qjl.as_ptr();
        let pairs = full_bytes / 2;
        let mut byte_idx = 0;
        while byte_idx < pairs * 2 {
            let m0 = SIGN_MASK_TABLE[*s2.get_unchecked(byte_idx) as usize];
            let m1 = SIGN_MASK_TABLE[*s2.get_unchecked(byte_idx + 1) as usize];
            let base = byte_idx * 8;

            let q0 = vld1q_f32(qptr.add(base));
            let q1 = vld1q_f32(qptr.add(base + 4));
            let q2 = vld1q_f32(qptr.add(base + 8));
            let q3 = vld1q_f32(qptr.add(base + 12));

            let mv0 = vld1q_u32(m0.as_ptr());
            let mv1 = vld1q_u32(m0.as_ptr().add(4));
            let mv2 = vld1q_u32(m1.as_ptr());
            let mv3 = vld1q_u32(m1.as_ptr().add(4));

            let f0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q0), mv0));
            let f1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q1), mv1));
            let f2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q2), mv2));
            let f3 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q3), mv3));

            acc0 = vaddq_f32(acc0, f0);
            acc1 = vaddq_f32(acc1, f1);
            acc2 = vaddq_f32(acc2, f2);
            acc3 = vaddq_f32(acc3, f3);

            byte_idx += 2;
        }
        // Remaining single byte (when full_bytes is odd).
        if byte_idx < full_bytes {
            let m = SIGN_MASK_TABLE[*s2.get_unchecked(byte_idx) as usize];
            let base = byte_idx * 8;
            let q0 = vld1q_f32(qptr.add(base));
            let q1 = vld1q_f32(qptr.add(base + 4));
            let mv0 = vld1q_u32(m.as_ptr());
            let mv1 = vld1q_u32(m.as_ptr().add(4));
            let f0 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q0), mv0));
            let f1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(q1), mv1));
            acc0 = vaddq_f32(acc0, f0);
            acc1 = vaddq_f32(acc1, f1);
        }

        let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(s)
    }

    /// Stage 1 for fixed **5-bit** TurboQuant (4-bit stage-1 indices per slot).
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn stage1_dot_b5_neon(
        s1: &[u8],
        rotated_q: &[f32],
        cb: &[f32; 16],
        d: usize,
    ) -> f32 {
        debug_assert_eq!(d % 16, 0, "stage1_dot_b5_neon expects d divisible by 16");
        debug_assert!(s1.len() * 2 >= d, "stage1_dot_b5_neon: s1 too short");

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let qptr = rotated_q.as_ptr();
        let cb_ptr = cb.as_ptr();

        for chunk in 0..(d / 16) {
            let s_off = chunk * 8;
            let base = chunk * 16;
            let b0 = *s1.get_unchecked(s_off);
            let b1 = *s1.get_unchecked(s_off + 1);
            let b2 = *s1.get_unchecked(s_off + 2);
            let b3 = *s1.get_unchecked(s_off + 3);
            let b4 = *s1.get_unchecked(s_off + 4);
            let b5 = *s1.get_unchecked(s_off + 5);
            let b6 = *s1.get_unchecked(s_off + 6);
            let b7 = *s1.get_unchecked(s_off + 7);

            let cv0 = [
                *cb_ptr.add((b0 >> 4) as usize),
                *cb_ptr.add((b0 & 0x0F) as usize),
                *cb_ptr.add((b1 >> 4) as usize),
                *cb_ptr.add((b1 & 0x0F) as usize),
            ];
            let cv1 = [
                *cb_ptr.add((b2 >> 4) as usize),
                *cb_ptr.add((b2 & 0x0F) as usize),
                *cb_ptr.add((b3 >> 4) as usize),
                *cb_ptr.add((b3 & 0x0F) as usize),
            ];
            let cv2 = [
                *cb_ptr.add((b4 >> 4) as usize),
                *cb_ptr.add((b4 & 0x0F) as usize),
                *cb_ptr.add((b5 >> 4) as usize),
                *cb_ptr.add((b5 & 0x0F) as usize),
            ];
            let cv3 = [
                *cb_ptr.add((b6 >> 4) as usize),
                *cb_ptr.add((b6 & 0x0F) as usize),
                *cb_ptr.add((b7 >> 4) as usize),
                *cb_ptr.add((b7 & 0x0F) as usize),
            ];

            let c0 = vld1q_f32(cv0.as_ptr());
            let c1 = vld1q_f32(cv1.as_ptr());
            let c2 = vld1q_f32(cv2.as_ptr());
            let c3 = vld1q_f32(cv3.as_ptr());
            let q0 = vld1q_f32(qptr.add(base));
            let q1 = vld1q_f32(qptr.add(base + 4));
            let q2 = vld1q_f32(qptr.add(base + 8));
            let q3 = vld1q_f32(qptr.add(base + 12));

            acc0 = vfmaq_f32(acc0, c0, q0);
            acc1 = vfmaq_f32(acc1, c1, q1);
            acc2 = vfmaq_f32(acc2, c2, q2);
            acc3 = vfmaq_f32(acc3, c3, q3);
        }

        let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(s)
    }
}

#[cfg(test)]
mod tests {
    use super::super::quantizer::TurboQuantizer;
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

    fn exact_ip(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Recall@k of turboquant vs brute-force exact IP on a random unit-sphere
    /// dataset. Tight enough to catch regressions in the end-to-end pipeline
    /// (encode + query + IP estimate + sort order).
    #[test]
    fn recall_at_10_beats_floor() {
        let d = 256;
        let n = 1000;
        let k_recall = 10;

        let tq = TurboQuantizer::new(d, Some(42));

        let docs: Vec<Vec<f32>> = (0..n).map(|i| unit_rand(d, 1_000 + i as u64)).collect();
        let records: Vec<Vec<u8>> = docs.iter().map(|v| tq.encode(v)).collect();

        let queries: Vec<Vec<f32>> = (0..10).map(|i| unit_rand(d, 7_000 + i)).collect();

        let mut total_recall = 0usize;
        for q in &queries {
            let mut exact: Vec<(usize, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, exact_ip(v, q)))
                .collect();
            exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let gt: std::collections::HashSet<usize> =
                exact.iter().take(k_recall).map(|(i, _)| *i).collect();

            let tqq = TurboQuantQuery::new(&tq, q);
            let mut est: Vec<(usize, f32)> = records
                .iter()
                .enumerate()
                .map(|(i, r)| (i, tqq.estimate_ip(r)))
                .collect();
            est.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top: std::collections::HashSet<usize> =
                est.iter().take(k_recall).map(|(i, _)| *i).collect();

            total_recall += gt.intersection(&top).count();
        }
        let avg_recall = total_recall as f32 / (queries.len() * k_recall) as f32;
        assert!(
            avg_recall >= 0.5,
            "recall@10 below sanity floor: {avg_recall}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_b5_matches_scalar() {
        let tq = TurboQuantizer::new(256, Some(42));
        let doc = unit_rand(256, 11);
        let query = unit_rand(256, 12);
        let record = tq.encode(&doc);
        let tqq = TurboQuantQuery::new(&tq, &query);
        let neon = tqq.estimate_ip(&record);
        let scalar = tqq.estimate_ip_scalar(&record);
        assert!(
            (neon - scalar).abs() < 1e-4,
            "neon={neon} scalar={scalar}"
        );
    }

    #[test]
    fn self_ip_close_to_one() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(42));
        let v = unit_rand(d, 1);
        let rec = tq.encode(&v);
        let q = TurboQuantQuery::new(&tq, &v);
        let ip = q.estimate_ip(&rec);
        assert!((ip - 1.0).abs() < 0.2, "self-IP too far from 1.0: {ip}");
    }

    #[test]
    fn neon_matches_scalar_roundtrip() {
        let d = 768;
        let tq = TurboQuantizer::new(d, Some(42));
        let docs: Vec<Vec<f32>> = (0..16).map(|i| unit_rand(d, 100 + i)).collect();
        let recs: Vec<Vec<u8>> = docs.iter().map(|v| tq.encode(v)).collect();
        let q = unit_rand(d, 9_001);
        let qq = TurboQuantQuery::new(&tq, &q);

        for rec in &recs {
            let scalar = qq.estimate_ip_scalar(rec);
            let combined = qq.estimate_ip(rec);
            assert!(
                (scalar - combined).abs() < 1e-4,
                "scalar {scalar} vs simd {combined}"
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_estimate_ip() {
        use std::time::Instant;

        let d = 768;
        let n = 60_000;

        let tq = TurboQuantizer::new(d, Some(42));
        let docs: Vec<Vec<u8>> = (0..n)
            .map(|i| tq.encode(&unit_rand(d, 1_000 + i as u64)))
            .collect();
        let q = unit_rand(d, 9_001);
        let qq = TurboQuantQuery::new(&tq, &q);

        let mut sink = 0.0f32;
        for r in &docs {
            sink += qq.estimate_ip(r);
        }

        let start = Instant::now();
        for r in &docs {
            sink += qq.estimate_ip(r);
        }
        let total = start.elapsed();
        eprintln!(
            "SIMD: {} docs in {:?}, {:.1} ns/doc, sum {sink}",
            n,
            total,
            total.as_nanos() as f64 / n as f64
        );

        let mut qq_scalar = TurboQuantQuery::new(&tq, &q);
        qq_scalar.use_simd = false;

        let start = Instant::now();
        for r in &docs {
            sink += qq_scalar.estimate_ip(r);
        }
        let total = start.elapsed();
        eprintln!(
            "Scalar: {} docs in {:?}, {:.1} ns/doc, sum {sink}",
            n,
            total,
            total.as_nanos() as f64 / n as f64
        );
    }
}
