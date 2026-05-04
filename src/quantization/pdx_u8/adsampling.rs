//! ADSampling rotation + pruning ratios (`include/pdx/pruners/adsampling.hpp`,
//! `ADSAMPLING_PRUNING_AGGRESIVENESS` in `common.hpp`).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

pub const ADSAMPLING_PRUNING_AGGRESSIVENESS: f32 = 1.5;

#[inline]
fn get_ratio(visited_dimensions: usize, num_dimensions: usize, pruning_aggressiveness: f32) -> f32 {
    if visited_dimensions == 0 || visited_dimensions == num_dimensions {
        return 1.0;
    }
    let vd = visited_dimensions as f32;
    let nd = num_dimensions as f32;
    let t = 1.0 + pruning_aggressiveness / vd.sqrt();
    (vd / nd) * t * t
}

/// Random orthogonal matrix as **column-orthonormal** columns stored column-major (`flat[col*d + row]`).
pub fn random_orthogonal_columns(dimensions: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).expect("normal");
    let mut a = vec![0.0f32; dimensions * dimensions];
    for j in 0..dimensions {
        for i in 0..dimensions {
            a[j * dimensions + i] = normal.sample(&mut rng);
        }
    }
    modified_gram_schmidt(&mut a, dimensions);
    a
}

/// In-place: columns of `a` become orthonormal (column-major layout, length `dimensions²`).
fn modified_gram_schmidt(a: &mut [f32], dimensions: usize) {
    let d = dimensions;
    let mut tmp = vec![0.0f32; d];
    let mut q = vec![0.0f32; d * d];
    for j in 0..d {
        for i in 0..d {
            tmp[i] = a[j * d + i];
        }
        for k in 0..j {
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += tmp[i] * q[k * d + i];
            }
            for i in 0..d {
                tmp[i] -= dot * q[k * d + i];
            }
        }
        let norm = tmp.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for i in 0..d {
            q[j * d + i] = tmp[i] / norm;
        }
    }
    a.copy_from_slice(&q);
}

/// `out = Q * x` with column-major orthogonal `q_flat`.
#[inline]
pub fn rotation_matvec(q_flat: &[f32], x: &[f32], out: &mut [f32]) {
    let d = x.len();
    debug_assert_eq!(q_flat.len(), d * d);
    debug_assert_eq!(out.len(), d);
    out.fill(0.0);
    for j in 0..d {
        let xj = x[j];
        let base = j * d;
        for i in 0..d {
            out[i] += q_flat[base + i] * xj;
        }
    }
}

pub struct AdsamplingPruner {
    pub num_dimensions: usize,
    rotation_col_major: Vec<f32>,
    ratios: Vec<f32>,
}

impl AdsamplingPruner {
    pub fn new(num_dimensions: usize, seed: u64) -> Self {
        let ratios = (0..num_dimensions)
            .map(|i| get_ratio(i, num_dimensions, ADSAMPLING_PRUNING_AGGRESSIVENESS))
            .collect::<Vec<_>>();
        Self {
            num_dimensions,
            rotation_col_major: random_orthogonal_columns(num_dimensions, seed),
            ratios,
        }
    }

    #[inline]
    pub fn preprocess_query(&self, src: &[f32], out: &mut [f32]) {
        rotation_matvec(&self.rotation_col_major, src, out);
    }

    /// Same rotation as [`Self::preprocess_query`], for encoding corpus vectors into the ADSampling basis.
    #[inline]
    pub fn rotate_embedding(&self, src: &[f32], out: &mut [f32]) {
        rotation_matvec(&self.rotation_col_major, src, out);
    }

    /// PDX `GetPruningThreshold` factor before multiplying heap worst distance.
    #[inline]
    pub fn pruning_distance_ratio(&self, current_dimension_idx: u32) -> f32 {
        if current_dimension_idx as usize >= self.num_dimensions {
            return 1.0;
        }
        self.ratios[current_dimension_idx as usize]
    }
}
