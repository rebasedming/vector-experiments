//! Lifted FAISS-style sampled Lloyd's k-means from
//! [tantivy/src/vector/cluster/kmeans.rs](https://github.com/quickwit-oss/tantivy)
//! (turboquant branch).
//!
//! Algorithmically identical to the upstream — same FAISS-style sampling,
//! same farthest-point empty-cluster reseeding, same `nredo` best-of-N
//! retries. Only the BLAS dispatch is rerouted through our shared
//! [`crate::kmeans::superkmeans::blas`] abstraction so it picks up Apple
//! Accelerate / BLIS instead of `matrixmultiply` directly. That makes the
//! variant a fair head-to-head against `superkmeans` on whatever hardware
//! the experiment runs on.

pub mod algo;

pub use algo::{run_kmeans, run_kmeans_with_config, KMeansConfig, KMeansResult};
