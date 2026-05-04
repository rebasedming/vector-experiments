//! ADSampling pruner: random rotation matrix and per-dimension pruning ratios.
//!
//! Mirrors `include/superkmeans/pdx/adsampling.h`. The DCT/FFTW path is
//! intentionally omitted; this port always uses the random orthogonal matrix
//! built from QR of a Gaussian draw.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use super::blas::{sgemm_row_major, Trans};

/// Random orthogonal rotation + ADSampling pruning ratio cache.
pub struct AdSamplingPruner {
    pub num_dimensions: u32,
    pub ratios: Vec<f32>,
    epsilon0: f32,
    /// Row-major `d × d` orthogonal matrix.
    matrix: Vec<f32>,
}

impl AdSamplingPruner {
    pub fn new(num_dimensions: u32, epsilon0: f32, seed: u64) -> Self {
        let d = num_dimensions as usize;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = vec![0.0f32; d * d];
        let normal = StandardNormal;
        for value in matrix.iter_mut() {
            let v: f64 = normal.sample(&mut rng);
            *value = v as f32;
        }
        householder_qr_orthogonalize_in_place(&mut matrix, d);

        let mut pruner = Self {
            num_dimensions,
            ratios: Vec::new(),
            epsilon0,
            matrix,
        };
        pruner.initialize_ratios();
        pruner
    }

    pub fn set_epsilon0(&mut self, eps0: f32) {
        self.epsilon0 = eps0;
        self.initialize_ratios();
    }

    fn initialize_ratios(&mut self) {
        let d = self.num_dimensions as usize;
        let eps0 = self.epsilon0;
        self.ratios.clear();
        self.ratios.resize(d + 1, 0.0);
        for (i, slot) in self.ratios.iter_mut().enumerate() {
            *slot = compute_ratio(eps0, d, i);
        }
    }

    /// `best.distance * ratios[current_dim_idx]`.
    pub fn pruning_threshold(&self, best: &super::common::KnnCandidate, current_dim_idx: u32) -> f32 {
        best.distance * self.ratios[current_dim_idx as usize]
    }

    /// `out = data @ Q^T`, faithful to the C++ `sgemm_("T", "N", ...)` call.
    pub fn rotate(&self, data: &[f32], out: &mut [f32], n: usize) {
        let d = self.num_dimensions as usize;
        debug_assert_eq!(data.len(), n * d);
        debug_assert_eq!(out.len(), n * d);
        sgemm_row_major(
            Trans::No,
            Trans::Yes,
            n,
            d,
            d,
            1.0,
            data,
            d,
            &self.matrix,
            d,
            0.0,
            out,
            d,
        );
    }

    /// Inverse of `rotate`: `out = data @ Q`.
    pub fn unrotate(&self, data: &[f32], out: &mut [f32], n: usize) {
        let d = self.num_dimensions as usize;
        debug_assert_eq!(data.len(), n * d);
        debug_assert_eq!(out.len(), n * d);
        sgemm_row_major(
            Trans::No,
            Trans::No,
            n,
            d,
            d,
            1.0,
            data,
            d,
            &self.matrix,
            d,
            0.0,
            out,
            d,
        );
    }
}

fn compute_ratio(epsilon0: f32, num_dimensions: usize, visited_dimensions: usize) -> f32 {
    if visited_dimensions == 0 || visited_dimensions == num_dimensions {
        return 1.0;
    }
    let visited = visited_dimensions as f64;
    let total = num_dimensions as f64;
    let factor = 1.0 + epsilon0 as f64 / visited.sqrt();
    (visited / total * factor * factor) as f32
}

/// Replace `m` (a `d × d` row-major matrix) with the orthogonal `Q` factor of
/// its QR decomposition. Modified Gram-Schmidt with Householder-style
/// reflections; matches what Eigen's `HouseholderQR(...).householderQ() * I`
/// produces up to sign conventions.
fn householder_qr_orthogonalize_in_place(m: &mut [f32], d: usize) {
    debug_assert_eq!(m.len(), d * d);
    // Work in column-major scratch for clarity. Treat input rows as
    // `d` column vectors of length `d`. Then orthonormalize by Modified
    // Gram-Schmidt and write back.
    let mut cols: Vec<Vec<f64>> = (0..d)
        .map(|j| (0..d).map(|i| m[i * d + j] as f64).collect())
        .collect();

    for j in 0..d {
        // Subtract projections onto previously orthonormalized columns.
        for k in 0..j {
            let mut dot = 0.0f64;
            for i in 0..d {
                dot += cols[k][i] * cols[j][i];
            }
            for i in 0..d {
                cols[j][i] -= dot * cols[k][i];
            }
        }
        // Normalize.
        let mut norm_sq = 0.0f64;
        for i in 0..d {
            norm_sq += cols[j][i] * cols[j][i];
        }
        let inv = if norm_sq > 0.0 {
            1.0 / norm_sq.sqrt()
        } else {
            // Degenerate column: set to a canonical basis vector.
            for v in cols[j].iter_mut() {
                *v = 0.0;
            }
            cols[j][j] = 1.0;
            1.0
        };
        for i in 0..d {
            cols[j][i] *= inv;
        }
    }

    for i in 0..d {
        for j in 0..d {
            m[i * d + j] = cols[j][i] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn ratios_are_one_at_endpoints() {
        let p = AdSamplingPruner::new(64, 1.5, 42);
        assert!(approx(p.ratios[0], 1.0, 1e-6));
        assert!(approx(p.ratios[64], 1.0, 1e-6));
    }

    #[test]
    fn matrix_is_orthogonal() {
        let p = AdSamplingPruner::new(32, 1.5, 42);
        let d = 32;
        for i in 0..d {
            for j in 0..d {
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += p.matrix[k * d + i] * p.matrix[k * d + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-3,
                    "Q^T Q at ({i},{j}) = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rotation_preserves_l2_norm() {
        let p = AdSamplingPruner::new(48, 1.5, 7);
        let mut rng = StdRng::seed_from_u64(123);
        let n = 5usize;
        let d = 48usize;
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut rotated = vec![0.0; n * d];
        p.rotate(&data, &mut rotated, n);
        for i in 0..n {
            let orig: f32 = data[i * d..(i + 1) * d].iter().map(|v| v * v).sum();
            let rot: f32 = rotated[i * d..(i + 1) * d].iter().map(|v| v * v).sum();
            assert!(
                (orig - rot).abs() < 1e-2,
                "row {i}: orig={orig}, rot={rot}"
            );
        }
    }

    #[test]
    fn rotate_then_unrotate_roundtrips() {
        let p = AdSamplingPruner::new(32, 1.5, 7);
        let mut rng = StdRng::seed_from_u64(123);
        let n = 4usize;
        let d = 32usize;
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut rotated = vec![0.0; n * d];
        let mut roundtrip = vec![0.0; n * d];
        p.rotate(&data, &mut rotated, n);
        p.unrotate(&rotated, &mut roundtrip, n);
        for (i, (a, b)) in data.iter().zip(roundtrip.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "i={i}: data={a}, roundtrip={b}"
            );
        }
    }
}
