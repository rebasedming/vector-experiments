//! Shared constants and types for the SuperKMeans port.
//!
//! Mirrors `include/superkmeans/common.h`. Faithful 1:1 port for the
//! `(Quantization::f32, DistanceFunction::l2)` instantiation only.

pub const PROPORTION_HORIZONTAL_DIM: f64 = 0.75;
pub const D_THRESHOLD_FOR_DCT_ROTATION: usize = 512;
pub const H_DIM_SIZE: usize = 64;

pub const MIN_PARTIAL_D: u32 = 16;

pub const DIMENSION_THRESHOLD_FOR_PRUNING: usize = 128;
pub const N_CLUSTERS_THRESHOLD_FOR_PRUNING: usize = 256;

#[cfg(target_os = "macos")]
pub const X_BATCH_SIZE: usize = 40_960;
#[cfg(target_os = "macos")]
pub const Y_BATCH_SIZE: usize = 2_048;

#[cfg(not(target_os = "macos"))]
pub const X_BATCH_SIZE: usize = 4_096;
#[cfg(not(target_os = "macos"))]
pub const Y_BATCH_SIZE: usize = 1_024;

/// Mini-batch chunk used inside the Apple GEMM dispatch (matches the C++
/// `#if defined(__APPLE__)` AMX-friendly partitioning).
pub const MINI_BATCH_SIZE: usize = 256;

pub const VECTOR_CHUNK_SIZE: usize = Y_BATCH_SIZE;

pub const RECALL_CONVERGENCE_PATIENCE: usize = 2;
pub const CENTROID_PERTURBATION_EPS: f32 = 1.0 / 1024.0;
pub const PRUNER_INITIAL_THRESHOLD: f32 = 1.5;
pub const HIERARCHICAL_PRUNER_INITIAL_THRESHOLD: f32 = 1.1;

pub fn align_value(n: usize, val: usize) -> usize {
    n.div_ceil(val) * val
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceFunction {
    L2,
    Dp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    F32,
}

#[derive(Debug, Clone, Copy)]
pub struct KnnCandidate {
    pub index: u32,
    pub distance: f32,
}

impl Default for KnnCandidate {
    fn default() -> Self {
        Self {
            index: 0,
            distance: f32::MAX,
        }
    }
}

/// One per-block (`VECTOR_CHUNK_SIZE` rows) view into the PDX data.
#[derive(Clone, Copy)]
pub struct ClusterView {
    pub num_embeddings: u32,
    pub data_offset: usize,
    pub indices_offset: usize,
    pub aux_offset: usize,
}
