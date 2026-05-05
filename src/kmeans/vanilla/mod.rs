//! Vanilla copy of `tantivy/src/vector/cluster/kmeans.rs` (turboquant branch),
//! lifted verbatim — including its direct `matrixmultiply::sgemm` BLAS path.

pub mod algo;

pub use algo::{run_kmeans, run_kmeans_with_config, KMeansConfig, KMeansResult};
