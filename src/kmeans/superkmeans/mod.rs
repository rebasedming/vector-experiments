//! Faithful 1:1 Rust port of [cwida/SuperKMeans](https://github.com/cwida/SuperKMeans)
//! for the `(Quantization::f32, DistanceFunction::l2)` instantiation.
//!
//! Deliberate omissions from the C++ source:
//! * `Quantization::u8` and `DistanceFunction::dp` template variants.
//! * `HierarchicalSuperKMeans`.
//! * GPU paths.
//! * AVX2 / AVX-512 / NEON intrinsic kernels — we use the scalar fallback
//!   from `scalar_computers.h` everywhere.
//! * The optional FFTW DCT-based rotation — we always use the random
//!   orthogonal matrix from QR of a Gaussian draw, which is the C++ fallback
//!   when FFTW is absent.
//! * `Profiler` — replaced by simple verbose logging.

pub mod algo;
pub mod batch;
pub mod blas;
pub mod common;
pub mod computers;
pub mod layout;
pub mod pdxearch;
pub mod rotation;

pub use algo::{
    ClusterBalanceStats, SuperKMeans, SuperKMeansConfig, SuperKMeansIterationStats,
};
pub use common::{DistanceFunction, KnnCandidate, Quantization};
