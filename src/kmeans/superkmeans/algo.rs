//! `SuperKMeans` main class.
//!
//! Mirrors `include/superkmeans/superkmeans.h` for the
//! `(Quantization::f32, DistanceFunction::l2)` instantiation.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::batch::{
    find_k_nearest_neighbors, find_nearest_neighbor, find_nearest_neighbor_with_pruning,
    PdxCentroidsRef,
};
use super::common::{
    KnnCandidate, CENTROID_PERTURBATION_EPS, DIMENSION_THRESHOLD_FOR_PRUNING, MIN_PARTIAL_D,
    N_CLUSTERS_THRESHOLD_FOR_PRUNING, PRUNER_INITIAL_THRESHOLD, RECALL_CONVERGENCE_PATIENCE,
    X_BATCH_SIZE, Y_BATCH_SIZE,
};
use super::layout::{get_dimension_split, pdxify, IndexPdxIvf};
use super::rotation::AdSamplingPruner;

#[derive(Debug, Clone)]
pub struct SuperKMeansConfig {
    pub iters: u32,
    pub sampling_fraction: f32,
    pub max_points_per_cluster: u32,
    pub n_threads: u32,
    pub seed: u64,
    pub use_blas_only: bool,
    pub tol: f32,
    pub recall_tol: f32,
    pub early_termination: bool,
    pub sample_queries: bool,
    pub objective_k: usize,
    pub ann_explore_fraction: f32,
    pub min_not_pruned_pct: f32,
    pub max_not_pruned_pct: f32,
    pub adjustment_factor_for_partial_d: f32,
    pub unrotate_centroids: bool,
    pub verbose: bool,
    pub angular: bool,
    pub suppress_warnings: bool,
    pub data_already_rotated: bool,
}

impl Default for SuperKMeansConfig {
    fn default() -> Self {
        Self {
            iters: 10,
            sampling_fraction: 0.3,
            max_points_per_cluster: 256,
            n_threads: 0,
            seed: 42,
            use_blas_only: false,
            tol: 1e-4,
            recall_tol: 0.005,
            early_termination: true,
            sample_queries: false,
            objective_k: 100,
            ann_explore_fraction: 0.01,
            min_not_pruned_pct: 0.03,
            max_not_pruned_pct: 0.05,
            adjustment_factor_for_partial_d: 0.20,
            unrotate_centroids: true,
            verbose: false,
            angular: false,
            suppress_warnings: false,
            data_already_rotated: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SuperKMeansIterationStats {
    pub iteration: usize,
    pub objective: f32,
    pub shift: f32,
    pub split: usize,
    pub recall: f32,
    pub not_pruned_pct: f32,
    pub partial_d: u32,
    pub is_gemm_only: bool,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ClusterBalanceStats {
    pub mean: f32,
    pub geometric_mean: f32,
    pub stdev: f32,
    pub cv: f32,
    pub min: usize,
    pub max: usize,
}

pub struct SuperKMeans {
    pub d: usize,
    pub n_clusters: usize,
    pub config: SuperKMeansConfig,

    n_threads: u32,
    n_samples: usize,
    partial_d: u32,
    vertical_d: u32,

    pruner: AdSamplingPruner,

    // Centroid buffers (all in rotated space until output).
    centroids_pdx: Vec<f32>,
    centroids_pdx_indices: Vec<u32>,
    horizontal_centroids: Vec<f32>,
    prev_centroids: Vec<f32>,
    partial_horizontal_centroids: Vec<f32>,
    pdx_index: Option<IndexPdxIvf>,

    // Per-sample state.
    pub assignments: Vec<u32>,
    distances: Vec<f32>,
    cluster_sizes: Vec<u32>,
    data_norms: Vec<f32>,
    data_norms_are_partial: bool,
    centroid_norms: Vec<f32>,
    sampled_indices: Vec<usize>,

    // Recall / GT state (only allocated if queries provided).
    gt_assignments: Vec<u32>,
    gt_distances: Vec<f32>,
    query_norms: Vec<f32>,
    promising_centroids: Vec<u32>,
    recall_distances: Vec<f32>,
    centroids_to_explore: usize,

    // Iteration tracking.
    pub iteration_stats: Vec<SuperKMeansIterationStats>,
    pub trained: bool,
    n_split: usize,
    prev_cost: f32,
    cost: f32,
    shift: f32,
    recall: f32,
}

impl SuperKMeans {
    pub fn new(n_clusters: usize, dimensionality: usize, config: SuperKMeansConfig) -> Self {
        assert!(n_clusters > 0, "n_clusters must be positive");
        assert!(dimensionality > 0, "dimensionality must be positive");
        assert!(config.iters > 0, "iters must be positive");
        assert!(
            config.sampling_fraction > 0.0,
            "sampling_fraction must be positive"
        );
        assert!(
            config.sampling_fraction <= 1.0,
            "sampling_fraction must be <= 1.0"
        );

        let n_threads = if config.n_threads == 0 {
            rayon::current_num_threads() as u32
        } else {
            config.n_threads
        };

        let pruner = AdSamplingPruner::new(
            dimensionality as u32,
            PRUNER_INITIAL_THRESHOLD,
            config.seed,
        );

        let mut config = config;
        if config.data_already_rotated {
            config.unrotate_centroids = false;
        }

        Self {
            d: dimensionality,
            n_clusters,
            config,
            n_threads,
            n_samples: 0,
            partial_d: 0,
            vertical_d: 0,
            pruner,
            centroids_pdx: Vec::new(),
            centroids_pdx_indices: Vec::new(),
            horizontal_centroids: Vec::new(),
            prev_centroids: Vec::new(),
            partial_horizontal_centroids: Vec::new(),
            pdx_index: None,
            assignments: Vec::new(),
            distances: Vec::new(),
            cluster_sizes: Vec::new(),
            data_norms: Vec::new(),
            data_norms_are_partial: false,
            centroid_norms: Vec::new(),
            sampled_indices: Vec::new(),
            gt_assignments: Vec::new(),
            gt_distances: Vec::new(),
            query_norms: Vec::new(),
            promising_centroids: Vec::new(),
            recall_distances: Vec::new(),
            centroids_to_explore: 0,
            iteration_stats: Vec::new(),
            trained: false,
            n_split: 0,
            prev_cost: 0.0,
            cost: 0.0,
            shift: 0.0,
            recall: 0.0,
        }
    }

    pub fn with_default_config(n_clusters: usize, dimensionality: usize) -> Self {
        Self::new(n_clusters, dimensionality, SuperKMeansConfig::default())
    }

    /// `Train` from the C++ source. Returns the (optionally unrotated) row-major
    /// centroid matrix of shape `n_clusters × d`.
    pub fn train(
        &mut self,
        data: &[f32],
        n: usize,
        queries: Option<&[f32]>,
        n_queries: usize,
    ) -> Vec<f32> {
        assert!(n > 0);
        assert!(!self.trained, "already trained");
        self.iteration_stats.clear();
        assert!(
            n >= self.n_clusters,
            "n must be at least as large as n_clusters",
        );
        if n_queries > 0 && queries.is_none() && !self.config.sample_queries {
            panic!("queries must be provided if n_queries > 0 and sample_queries is false");
        }

        self.n_samples = self.compute_n_samples(n, self.n_clusters);
        assert!(
            self.n_samples >= self.n_clusters,
            "not enough samples to train",
        );

        let d = self.d;
        let n_clusters = self.n_clusters;

        // Allocate per-sample buffers.
        self.horizontal_centroids = vec![0.0; n_clusters * d];
        self.prev_centroids = vec![0.0; n_clusters * d];
        self.cluster_sizes = vec![0; n_clusters];
        self.assignments = vec![0; n];
        self.distances = vec![0.0; n];
        self.data_norms = vec![0.0; self.n_samples];
        self.data_norms_are_partial = false;
        self.centroid_norms = vec![0.0; n_clusters];

        let split = get_dimension_split(d);
        self.vertical_d = split.vertical_d as u32;
        self.partial_horizontal_centroids = vec![0.0; n_clusters * split.vertical_d];

        self.partial_d = MIN_PARTIAL_D.max(split.vertical_d as u32 / 2);
        if self.partial_d > split.vertical_d as u32 {
            self.partial_d = split.vertical_d as u32;
        }

        // Build the IndexPdxIvf for the centroids and allocate the PDX buffers.
        let pdx_index = IndexPdxIvf::new(n_clusters, d);
        self.centroids_pdx = vec![0.0; pdx_index.data_buffer_len()];
        self.centroids_pdx_indices = (0..pdx_index.indices_buffer_len() as u32).collect();
        self.pdx_index = Some(pdx_index);

        // Generate initial centroids (Forgy sampling), rotate them, PDXify them.
        self.generate_centroids(data, self.n_samples, !self.config.data_already_rotated);

        // Sample + rotate the training data.
        let data_samples = self.sample_and_rotate_vectors(
            data,
            n,
            self.n_samples,
            !self.config.data_already_rotated,
        );

        // Rotate (or copy) the initial unrotated centroids in horizontal_centroids
        // into prev_centroids. (`horizontal_centroids` continues to hold the
        // unrotated centroids until the first FirstAssignAndUpdate zeroes it.)
        if self.config.data_already_rotated {
            self.prev_centroids
                .copy_from_slice(&self.horizontal_centroids[..n_clusters * d]);
        } else {
            self.pruner.rotate(
                &self.horizontal_centroids,
                &mut self.prev_centroids,
                n_clusters,
            );
        }

        // Initial norms (full).
        get_l2_norms_row_major(&data_samples, self.n_samples, d, &mut self.data_norms);
        get_l2_norms_row_major(
            &self.prev_centroids,
            n_clusters,
            d,
            &mut self.centroid_norms,
        );

        // Optional query / GT setup.
        let mut rotated_queries: Vec<f32> = Vec::new();
        if n_queries > 0 {
            self.centroids_to_explore =
                ((n_clusters as f32 * self.config.ann_explore_fraction) as usize).max(1);
            self.gt_assignments = vec![0; n_queries * self.config.objective_k];
            self.gt_distances = vec![0.0; n_queries * self.config.objective_k];
            self.promising_centroids = vec![0; n_queries * self.centroids_to_explore];
            self.recall_distances = vec![0.0; n_queries * self.centroids_to_explore];
            self.query_norms = vec![0.0; n_queries];
            rotated_queries = vec![0.0; n_queries * d];
            if self.config.sample_queries {
                let sub_data = self.sample_and_rotate_vectors(
                    &data_samples,
                    self.n_samples,
                    n_queries,
                    false,
                );
                rotated_queries.copy_from_slice(&sub_data[..n_queries * d]);
            } else if let Some(q_in) = queries {
                self.rotate_or_copy(
                    q_in,
                    &mut rotated_queries,
                    n_queries,
                    !self.config.data_already_rotated,
                );
            }
            get_l2_norms_row_major(
                &rotated_queries,
                n_queries,
                d,
                &mut self.query_norms,
            );
            self.compute_gt_assignments_and_distances(&data_samples, &rotated_queries, n_queries);
        }

        let always_gemm_only = d < DIMENSION_THRESHOLD_FOR_PRUNING
            || self.config.use_blas_only
            || n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING;
        let mut partial_norms_computed = false;
        let mut best_recall = 0.0f32;
        let mut iters_without_improvement = 0usize;

        let mut tmp_distances_buf: Vec<f32> = vec![0.0; X_BATCH_SIZE * Y_BATCH_SIZE];
        let mut centroids_partial_norms: Vec<f32> = vec![0.0; n_clusters];
        let mut not_pruned_counts: Vec<usize> = vec![0; self.n_samples];

        for iter_idx in 0..self.config.iters as usize {
            let use_gemm_only = iter_idx == 0 || always_gemm_only;
            if !use_gemm_only && !partial_norms_computed {
                get_partial_l2_norms_row_major(
                    &data_samples,
                    self.n_samples,
                    d,
                    self.partial_d as usize,
                    &mut self.data_norms,
                );
                self.data_norms_are_partial = true;
                partial_norms_computed = true;
            }
            self.run_iteration(
                use_gemm_only,
                &data_samples,
                &mut tmp_distances_buf,
                &mut centroids_partial_norms,
                &mut not_pruned_counts,
                if n_queries > 0 {
                    Some(&rotated_queries)
                } else {
                    None
                },
                n_queries,
                self.n_samples,
                n_clusters,
                iter_idx,
            );
            if self.config.early_termination
                && self.should_stop_early(
                    n_queries > 0,
                    &mut best_recall,
                    &mut iters_without_improvement,
                    iter_idx,
                )
            {
                break;
            }
        }
        self.trained = true;
        self.get_output_centroids(self.config.unrotate_centroids)
    }

    /// Brute-force assignment: bipartite L2 nearest neighbor search.
    pub fn assign(
        &self,
        vectors: &[f32],
        centroids: &[f32],
        n_vectors: usize,
        n_centroids: usize,
    ) -> Vec<u32> {
        let d = self.d;
        let mut result = vec![0u32; n_vectors];
        let mut tmp = vec![0.0; X_BATCH_SIZE * Y_BATCH_SIZE];
        let mut v_norms = vec![0.0; n_vectors];
        let mut c_norms = vec![0.0; n_centroids];
        let mut distances = vec![0.0; n_vectors];

        get_l2_norms_row_major(vectors, n_vectors, d, &mut v_norms);
        get_l2_norms_row_major(centroids, n_centroids, d, &mut c_norms);

        find_nearest_neighbor(
            vectors,
            centroids,
            n_vectors,
            n_centroids,
            d,
            &v_norms,
            &c_norms,
            &mut result,
            &mut distances,
            &mut tmp,
        );

        result
    }

    /// `AssignTrainingPoints`: leverages trained state to assign a corpus the
    /// pruning fast-path. Mirrors the three sampling-fraction branches.
    pub fn assign_training_points(
        &mut self,
        vectors: &[f32],
        centroids: &[f32],
        n_vectors: usize,
        n_centroids: usize,
    ) -> Vec<u32> {
        assert!(self.trained, "AssignTrainingPoints requires trained state");
        let d = self.d;
        if self.config.use_blas_only
            || d < DIMENSION_THRESHOLD_FOR_PRUNING
            || self.n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING
        {
            if !self.config.suppress_warnings {
                eprintln!(
                    "WARNING: AssignTrainingPoints cannot use pruning, falling back to brute force",
                );
            }
            return self.assign(vectors, centroids, n_vectors, n_centroids);
        }

        let mut result = vec![0u32; n_vectors];
        let mut tmp = vec![0.0; X_BATCH_SIZE * Y_BATCH_SIZE];

        self.partial_d = MIN_PARTIAL_D.max(self.vertical_d / 2);

        let mut not_pruned_counts = vec![0usize; n_vectors];

        let data_buffer: Vec<f32>;
        let data_p: &[f32] = if self.config.data_already_rotated {
            vectors
        } else {
            let mut buf = vec![0.0; n_vectors * d];
            self.pruner.rotate(vectors, &mut buf, n_vectors);
            data_buffer = buf;
            // Borrow extension: keep buffer alive via this pattern.
            // (We rebind below.)
            // Note: in Rust we can't return a temporary; use explicit branches.
            return self.assign_training_points_with_rotated_data(
                centroids,
                n_centroids,
                n_vectors,
                &data_buffer,
                &mut result,
                &mut tmp,
                &mut not_pruned_counts,
            );
        };

        self.assign_training_points_with_rotated_data(
            centroids,
            n_centroids,
            n_vectors,
            data_p,
            &mut result,
            &mut tmp,
            &mut not_pruned_counts,
        )
    }

    fn assign_training_points_with_rotated_data(
        &mut self,
        centroids: &[f32],
        n_centroids: usize,
        n_vectors: usize,
        data_p: &[f32],
        result: &mut Vec<u32>,
        tmp: &mut [f32],
        not_pruned_counts: &mut [usize],
    ) -> Vec<u32> {
        let d = self.d;

        // Refresh partial centroid norms.
        get_partial_l2_norms_row_major(
            &self.horizontal_centroids,
            n_centroids,
            d,
            self.partial_d as usize,
            &mut self.centroid_norms,
        );

        // Branch on sampling fraction (matches the three C++ paths).
        if self.config.sampling_fraction == 1.0 {
            // Recompute partial data norms over the full input.
            self.data_norms.resize(n_vectors, 0.0);
            get_partial_l2_norms_row_major(
                data_p,
                n_vectors,
                d,
                self.partial_d as usize,
                &mut self.data_norms,
            );
            self.data_norms_are_partial = true;

            let centroids_ref = PdxCentroidsRef {
                index: self.pdx_index.as_ref().expect("pdx_index initialised"),
                pdx_data: &self.centroids_pdx,
                pdx_indices: &self.centroids_pdx_indices,
                aux: &self.partial_horizontal_centroids,
            };
            find_nearest_neighbor_with_pruning(
                &self.pruner,
                data_p,
                &self.horizontal_centroids,
                n_vectors,
                n_centroids,
                d,
                &self.data_norms,
                &self.centroid_norms,
                &mut self.assignments,
                &mut self.distances,
                tmp,
                &centroids_ref,
                self.partial_d,
                not_pruned_counts,
            );
            result.copy_from_slice(&self.assignments[..n_vectors]);
            return std::mem::take(result);
        }

        if self.config.sampling_fraction > 0.8 {
            let mut cur = 0usize;
            while cur < self.n_samples {
                result[self.sampled_indices[cur]] = self.assignments[cur];
                cur += 1;
            }
            // Seed remaining vectors from the cluster_sizes distribution.
            let mut rng = StdRng::seed_from_u64(self.config.seed.wrapping_add(1));
            let total: u32 = self.cluster_sizes.iter().sum();
            while cur < n_vectors {
                let pick = rng.gen_range(0..total.max(1));
                let mut acc = 0u32;
                let mut chosen = 0u32;
                for (ci, &sz) in self.cluster_sizes.iter().enumerate() {
                    acc = acc.saturating_add(sz);
                    if pick < acc {
                        chosen = ci as u32;
                        break;
                    }
                }
                result[self.sampled_indices[cur]] = chosen;
                cur += 1;
            }
            self.data_norms.resize(n_vectors, 0.0);
            get_partial_l2_norms_row_major(
                data_p,
                n_vectors,
                d,
                self.partial_d as usize,
                &mut self.data_norms,
            );
            self.data_norms_are_partial = true;

            let centroids_ref = PdxCentroidsRef {
                index: self.pdx_index.as_ref().expect("pdx_index initialised"),
                pdx_data: &self.centroids_pdx,
                pdx_indices: &self.centroids_pdx_indices,
                aux: &self.partial_horizontal_centroids,
            };
            find_nearest_neighbor_with_pruning(
                &self.pruner,
                data_p,
                &self.horizontal_centroids,
                n_vectors,
                n_centroids,
                d,
                &self.data_norms,
                &self.centroid_norms,
                result,
                &mut self.distances,
                tmp,
                &centroids_ref,
                self.partial_d,
                not_pruned_counts,
            );
            return std::mem::take(result);
        }

        // sampling_fraction <= 0.8: meso-cluster bootstrap.
        let mut tmp_config = SuperKMeansConfig {
            iters: 10,
            sampling_fraction: 1.0,
            use_blas_only: false,
            verbose: self.config.verbose,
            suppress_warnings: self.config.suppress_warnings,
            seed: self.config.seed,
            angular: self.config.angular,
            data_already_rotated: self.config.data_already_rotated,
            ..SuperKMeansConfig::default()
        };
        // (pull in defaults for any field not overridden)
        let new_n_centroids = (n_centroids as f64).sqrt() as usize;
        tmp_config.iters = 10;
        let mut tmp_kmeans = SuperKMeans::new(new_n_centroids, d, tmp_config);
        let meso_centroids = tmp_kmeans.train(centroids, n_centroids, None, 0);
        let meso_assignments =
            tmp_kmeans.assign(centroids, &meso_centroids, n_centroids, new_n_centroids);
        let centroids_to_meso =
            tmp_kmeans.assign(centroids, &meso_centroids, n_centroids, new_n_centroids);
        let _ = meso_assignments; // suppress unused (matches structure of C++ which uses both)

        let mut meso_to_original = vec![0u32; new_n_centroids];
        for c in 0..n_centroids {
            meso_to_original[centroids_to_meso[c] as usize] = c as u32;
        }

        let mut cur = 0usize;
        while cur < self.n_samples {
            result[self.sampled_indices[cur]] = self.assignments[cur];
            cur += 1;
        }
        // Map non-sampled vectors via their meso-assignment to a representative original centroid.
        let vec_meso = tmp_kmeans.assign(data_p, &meso_centroids, n_vectors, new_n_centroids);
        while cur < n_vectors {
            let orig_idx = self.sampled_indices[cur];
            result[orig_idx] = meso_to_original[vec_meso[orig_idx] as usize];
            cur += 1;
        }

        self.data_norms.resize(n_vectors, 0.0);
        get_partial_l2_norms_row_major(
            data_p,
            n_vectors,
            d,
            self.partial_d as usize,
            &mut self.data_norms,
        );
        self.data_norms_are_partial = true;

        let centroids_ref = PdxCentroidsRef {
            index: self.pdx_index.as_ref().expect("pdx_index initialised"),
            pdx_data: &self.centroids_pdx,
            pdx_indices: &self.centroids_pdx_indices,
            aux: &self.partial_horizontal_centroids,
        };
        find_nearest_neighbor_with_pruning(
            &self.pruner,
            data_p,
            &self.horizontal_centroids,
            n_vectors,
            n_centroids,
            d,
            &self.data_norms,
            &self.centroid_norms,
            result,
            &mut self.distances,
            tmp,
            &centroids_ref,
            self.partial_d,
            not_pruned_counts,
        );
        std::mem::take(result)
    }

    pub fn cluster_balance_stats(
        assignments: &[u32],
        n_samples: usize,
        n_clusters: usize,
    ) -> ClusterBalanceStats {
        let mut sizes = vec![0usize; n_clusters];
        for &a in assignments.iter().take(n_samples) {
            sizes[a as usize] += 1;
        }
        let mean = sizes.iter().sum::<usize>() as f32 / n_clusters as f32;
        let mut log_sum = 0.0f32;
        let mut nz = 0usize;
        for &s in &sizes {
            if s > 0 {
                log_sum += (s as f32).ln();
                nz += 1;
            }
        }
        let geometric_mean = if nz > 0 {
            (log_sum / nz as f32).exp()
        } else {
            0.0
        };
        let sq: usize = sizes.iter().map(|s| s * s).sum();
        let variance = (sq as f32 / n_clusters as f32) - mean * mean;
        let stdev = variance.max(0.0).sqrt();
        let cv = if mean > 0.0 { stdev / mean } else { 0.0 };
        let min = *sizes.iter().min().unwrap_or(&0);
        let max = *sizes.iter().max().unwrap_or(&0);
        ClusterBalanceStats {
            mean,
            geometric_mean,
            stdev,
            cv,
            min,
            max,
        }
    }

    // ---- Internal helpers ----

    fn compute_n_samples(&self, n: usize, n_clusters: usize) -> usize {
        if self.config.sampling_fraction == 1.0 {
            return n;
        }
        let by_clusters = n_clusters * self.config.max_points_per_cluster as usize;
        let by_n = (n as f32 * self.config.sampling_fraction).floor() as usize;
        by_n.min(by_clusters)
    }

    fn rotate_or_copy(&self, src: &[f32], dst: &mut [f32], n_vectors: usize, rotate: bool) {
        if rotate {
            self.pruner.rotate(src, dst, n_vectors);
        } else {
            dst.copy_from_slice(&src[..n_vectors * self.d]);
        }
    }

    /// Forgy-style initial centroid sampling, mirrors `GenerateCentroids`.
    fn generate_centroids(&mut self, data: &[f32], n_points: usize, rotate: bool) {
        let d = self.d;
        let n_clusters = self.n_clusters;
        // Shuffle [0, n_points) and pick first n_clusters.
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut indices: Vec<usize> = (0..n_points).collect();
        indices.shuffle(&mut rng);
        for i in 0..n_clusters {
            let src_off = indices[i] * d;
            let dst_off = i * d;
            self.horizontal_centroids[dst_off..dst_off + d]
                .copy_from_slice(&data[src_off..src_off + d]);
        }

        // Rotate into a separate buffer, then PDXify.
        let mut rotated = vec![0.0; n_clusters * d];
        if rotate {
            self.pruner
                .rotate(&self.horizontal_centroids, &mut rotated, n_clusters);
        } else {
            rotated.copy_from_slice(&self.horizontal_centroids);
        }
        pdxify(&rotated, &mut self.centroids_pdx, n_clusters, d);
    }

    fn sample_and_rotate_vectors(
        &mut self,
        data: &[f32],
        n: usize,
        n_samples: usize,
        rotate: bool,
    ) -> Vec<f32> {
        let d = self.d;
        let mut out = vec![0.0; n_samples * d];

        if n_samples < n {
            let mut rng = StdRng::seed_from_u64(self.config.seed);
            self.sampled_indices = (0..n).collect();
            self.sampled_indices.shuffle(&mut rng);
            if rotate {
                let mut tmp = vec![0.0; n_samples * d];
                for i in 0..n_samples {
                    let src_off = self.sampled_indices[i] * d;
                    tmp[i * d..(i + 1) * d].copy_from_slice(&data[src_off..src_off + d]);
                }
                self.pruner.rotate(&tmp, &mut out, n_samples);
            } else {
                for i in 0..n_samples {
                    let src_off = self.sampled_indices[i] * d;
                    out[i * d..(i + 1) * d].copy_from_slice(&data[src_off..src_off + d]);
                }
            }
            return out;
        }

        // No sampling. The C++ optimises `!rotate && !sampled` to return the
        // input pointer; in Rust we materialise to keep ownership simple.
        if rotate {
            self.pruner.rotate(data, &mut out, n_samples);
        } else {
            out.copy_from_slice(&data[..n_samples * d]);
        }
        out
    }

    fn first_assign_and_update_centroids(
        &mut self,
        data: &[f32],
        rotated_initial_centroids: &[f32],
        tmp_distances_buf: &mut [f32],
        n_samples: usize,
        n_clusters: usize,
    ) {
        find_nearest_neighbor(
            data,
            rotated_initial_centroids,
            n_samples,
            n_clusters,
            self.d,
            &self.data_norms,
            &self.centroid_norms,
            &mut self.assignments,
            &mut self.distances,
            tmp_distances_buf,
        );
        self.horizontal_centroids.iter_mut().for_each(|v| *v = 0.0);
        self.cluster_sizes.iter_mut().for_each(|v| *v = 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_and_update_centroids(
        &mut self,
        data: &[f32],
        centroids: &[f32],
        partial_centroid_norms: &[f32],
        tmp_distances_buf: &mut [f32],
        not_pruned_counts: &mut [usize],
        n_samples: usize,
        n_clusters: usize,
    ) {
        let centroids_ref = PdxCentroidsRef {
            index: self.pdx_index.as_ref().expect("pdx_index initialised"),
            pdx_data: &self.centroids_pdx,
            pdx_indices: &self.centroids_pdx_indices,
            aux: &self.partial_horizontal_centroids,
        };
        find_nearest_neighbor_with_pruning(
            &self.pruner,
            data,
            centroids,
            n_samples,
            n_clusters,
            self.d,
            &self.data_norms,
            partial_centroid_norms,
            &mut self.assignments,
            &mut self.distances,
            tmp_distances_buf,
            &centroids_ref,
            self.partial_d,
            not_pruned_counts,
        );
        self.horizontal_centroids.iter_mut().for_each(|v| *v = 0.0);
        self.cluster_sizes.iter_mut().for_each(|v| *v = 0);
    }

    fn update_centroids(&mut self, data: &[f32], n_samples: usize, n_clusters: usize) {
        let d = self.d;
        let nt = (self.n_threads as usize).max(1);
        let chunk = n_clusters.div_ceil(nt).max(1);
        let centroid_chunks = self.horizontal_centroids.par_chunks_mut(chunk * d);
        let size_chunks = self.cluster_sizes.par_chunks_mut(chunk);
        centroid_chunks
            .zip(size_chunks)
            .enumerate()
            .for_each(|(slab_idx, (centroid_slab, sizes_slab))| {
                let c0 = slab_idx * chunk;
                let c1 = c0 + sizes_slab.len();
                for i in 0..n_samples {
                    let ci = self.assignments[i] as usize;
                    debug_assert!(ci < n_clusters);
                    if ci >= c0 && ci < c1 {
                        let local = ci - c0;
                        sizes_slab[local] += 1;
                        let dst = &mut centroid_slab[local * d..(local + 1) * d];
                        let src = &data[i * d..(i + 1) * d];
                        for j in 0..d {
                            dst[j] += src[j];
                        }
                    }
                }
            });
    }

    fn split_clusters(&mut self, n_samples: usize, n_clusters: usize) {
        self.n_split = 0;
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let d = self.d;
        let denom = (n_samples as i64 - n_clusters as i64).max(1) as f32;
        for ci in 0..n_clusters {
            if self.cluster_sizes[ci] != 0 {
                continue;
            }
            let mut cj: usize = 0;
            loop {
                let p = (self.cluster_sizes[cj] as f32 - 1.0) / denom;
                let r: f32 = rng.gen_range(0.0..1.0);
                if r < p {
                    break;
                }
                cj = (cj + 1) % n_clusters;
            }
            // Copy.
            let (lo, hi) = if ci < cj { (ci, cj) } else { (cj, ci) };
            let (lo_slice, hi_slice) = self.horizontal_centroids.split_at_mut(hi * d);
            let lo_part = &mut lo_slice[lo * d..lo * d + d];
            let hi_part = &mut hi_slice[..d];
            if ci < cj {
                lo_part.copy_from_slice(hi_part);
            } else {
                hi_part.copy_from_slice(lo_part);
            }
            // Symmetric perturbation.
            for j in 0..d {
                let ci_v = &mut self.horizontal_centroids[ci * d + j];
                if j % 2 == 0 {
                    *ci_v *= 1.0 + CENTROID_PERTURBATION_EPS;
                } else {
                    *ci_v *= 1.0 - CENTROID_PERTURBATION_EPS;
                }
            }
            for j in 0..d {
                let cj_v = &mut self.horizontal_centroids[cj * d + j];
                if j % 2 == 0 {
                    *cj_v *= 1.0 - CENTROID_PERTURBATION_EPS;
                } else {
                    *cj_v *= 1.0 + CENTROID_PERTURBATION_EPS;
                }
            }
            self.cluster_sizes[ci] = self.cluster_sizes[cj] / 2;
            self.cluster_sizes[cj] -= self.cluster_sizes[ci];
            self.n_split += 1;
        }
    }

    fn consolidate_centroids(&mut self, n_samples: usize, n_clusters: usize) {
        let d = self.d;
        for i in 0..n_clusters {
            if self.cluster_sizes[i] == 0 {
                continue;
            }
            let inv = 1.0 / self.cluster_sizes[i] as f32;
            for j in 0..d {
                self.horizontal_centroids[i * d + j] *= inv;
            }
        }
        self.split_clusters(n_samples, n_clusters);
        if self.config.angular {
            self.postprocess_centroids(n_clusters);
        }
        // PDXify horizontal_centroids → centroids_pdx.
        pdxify(&self.horizontal_centroids, &mut self.centroids_pdx, n_clusters, d);
        // Copy first vertical_d columns into partial_horizontal_centroids.
        let v = self.vertical_d as usize;
        for i in 0..n_clusters {
            let src_off = i * d;
            let dst_off = i * v;
            self.partial_horizontal_centroids[dst_off..dst_off + v]
                .copy_from_slice(&self.horizontal_centroids[src_off..src_off + v]);
        }
    }

    fn postprocess_centroids(&mut self, n_clusters: usize) {
        let d = self.d;
        for i in 0..n_clusters {
            let off = i * d;
            let mut sum = 0.0f32;
            for j in 0..d {
                let v = self.horizontal_centroids[off + j];
                sum += v * v;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum.sqrt();
                for j in 0..d {
                    self.horizontal_centroids[off + j] *= inv;
                }
            }
        }
    }

    fn compute_cost(&mut self, n_samples: usize) {
        self.prev_cost = self.cost;
        self.cost = self.distances[..n_samples].iter().sum();
    }

    fn compute_shift(&mut self, n_clusters: usize) {
        let d = self.d;
        let mut total = 0.0f32;
        for i in 0..n_clusters {
            let off = i * d;
            for j in 0..d {
                let diff = self.horizontal_centroids[off + j] - self.prev_centroids[off + j];
                total += diff * diff;
            }
        }
        self.shift = total;
    }

    fn compute_gt_assignments_and_distances(
        &mut self,
        data: &[f32],
        queries: &[f32],
        n_queries: usize,
    ) {
        let d = self.d;
        let mut gt_query_norms = vec![0.0; n_queries];
        get_l2_norms_row_major(queries, n_queries, d, &mut gt_query_norms);
        let mut tmp = vec![0.0; X_BATCH_SIZE * Y_BATCH_SIZE];
        find_k_nearest_neighbors(
            queries,
            data,
            n_queries,
            self.n_samples,
            d,
            &gt_query_norms,
            &self.data_norms,
            self.config.objective_k,
            &mut self.gt_assignments,
            &mut self.gt_distances,
            &mut tmp,
        );
    }

    fn compute_recall(&mut self, queries: &[f32], n_queries: usize) -> f32 {
        let d = self.d;
        let n_clusters = self.n_clusters;
        let mut tmp = vec![0.0; X_BATCH_SIZE * Y_BATCH_SIZE];
        find_k_nearest_neighbors(
            queries,
            &self.horizontal_centroids,
            n_queries,
            n_clusters,
            d,
            &self.query_norms,
            &self.centroid_norms,
            self.centroids_to_explore,
            &mut self.promising_centroids,
            &mut self.recall_distances,
            &mut tmp,
        );

        let mut sum_recall = 0.0f32;
        let k = self.config.objective_k;
        let to_explore = self.centroids_to_explore;
        for i in 0..n_queries {
            let mut found = 0usize;
            for j in 0..k {
                let gt_vec_idx = self.gt_assignments[i * k + j] as usize;
                let gt_centroid = self.assignments[gt_vec_idx];
                let mut hit = false;
                for t in 0..to_explore {
                    if self.promising_centroids[i * to_explore + t] == gt_centroid {
                        hit = true;
                        break;
                    }
                }
                if hit {
                    found += 1;
                }
            }
            sum_recall += found as f32 / k as f32;
        }
        sum_recall / n_queries.max(1) as f32
    }

    fn tune_partial_d(&mut self, not_pruned_counts: &[usize], n_samples: usize, n_y: usize) -> (f32, bool) {
        let mut avg = 0.0f32;
        for &v in not_pruned_counts.iter().take(n_samples) {
            avg += v as f32;
        }
        avg /= (n_samples * n_y).max(1) as f32;

        let old = self.partial_d;
        if avg > self.config.max_not_pruned_pct {
            let increase = ((self.partial_d as f32 * self.config.adjustment_factor_for_partial_d * 2.0)
                as u32)
                .max(1);
            self.partial_d = (self.partial_d + increase).min(self.vertical_d);
        } else if avg < self.config.min_not_pruned_pct {
            let decrease = ((self.partial_d as f32 * self.config.adjustment_factor_for_partial_d) as u32)
                .max(1);
            self.partial_d = self.partial_d.saturating_sub(decrease).max(MIN_PARTIAL_D);
        }
        (avg, old != self.partial_d)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_iteration(
        &mut self,
        gemm_only: bool,
        data_to_cluster: &[f32],
        tmp_distances_buf: &mut [f32],
        centroids_partial_norms: &mut [f32],
        not_pruned_counts: &mut [usize],
        rotated_queries: Option<&[f32]>,
        n_queries: usize,
        n_samples: usize,
        n_clusters: usize,
        iter_idx: usize,
    ) {
        let d = self.d;
        let is_first_iter = iter_idx == 0;
        let iter_start = std::time::Instant::now();
        if !is_first_iter {
            std::mem::swap(&mut self.horizontal_centroids, &mut self.prev_centroids);
        }

        if gemm_only {
            get_l2_norms_row_major(
                &self.prev_centroids,
                n_clusters,
                d,
                &mut self.centroid_norms,
            );
            // Move prev_centroids out temporarily; FirstAssign needs prev_centroids as &[f32].
            let prev = std::mem::take(&mut self.prev_centroids);
            self.first_assign_and_update_centroids(
                data_to_cluster,
                &prev,
                tmp_distances_buf,
                n_samples,
                n_clusters,
            );
            self.prev_centroids = prev;
        } else {
            get_partial_l2_norms_row_major(
                &self.prev_centroids,
                n_clusters,
                d,
                self.partial_d as usize,
                centroids_partial_norms,
            );
            not_pruned_counts.iter_mut().for_each(|v| *v = 0);
            let prev = std::mem::take(&mut self.prev_centroids);
            self.assign_and_update_centroids(
                data_to_cluster,
                &prev,
                centroids_partial_norms,
                tmp_distances_buf,
                not_pruned_counts,
                n_samples,
                n_clusters,
            );
            self.prev_centroids = prev;
        }

        self.update_centroids(data_to_cluster, n_samples, n_clusters);

        let mut avg_not_pruned = -1.0f32;
        let old_partial_d = self.partial_d;
        if !gemm_only {
            let (avg, changed) = self.tune_partial_d(not_pruned_counts, n_samples, n_clusters);
            avg_not_pruned = avg;
            if changed {
                get_partial_l2_norms_row_major(
                    data_to_cluster,
                    n_samples,
                    d,
                    self.partial_d as usize,
                    &mut self.data_norms,
                );
                self.data_norms_are_partial = true;
            }
        }

        self.consolidate_centroids(n_samples, n_clusters);
        self.compute_cost(n_samples);
        self.compute_shift(n_clusters);

        if n_queries > 0 {
            get_l2_norms_row_major(
                &self.horizontal_centroids,
                n_clusters,
                d,
                &mut self.centroid_norms,
            );
            self.recall = self.compute_recall(rotated_queries.unwrap(), n_queries);
        }

        let stats = SuperKMeansIterationStats {
            iteration: iter_idx + 1,
            objective: self.cost,
            shift: self.shift,
            split: self.n_split,
            recall: self.recall,
            is_gemm_only: gemm_only,
            not_pruned_pct: if gemm_only { -1.0 } else { avg_not_pruned },
            partial_d: if gemm_only { 0 } else { old_partial_d },
            duration_ms: iter_start.elapsed().as_secs_f64() * 1000.0,
        };
        self.iteration_stats.push(stats);

        if self.config.verbose {
            let improvement = if iter_idx > 0 {
                1.0 - (self.cost / self.prev_cost)
            } else {
                0.0
            };
            if gemm_only {
                eprintln!(
                    "Iteration {}/{} | Objective: {} | Improvement: {} | Shift: {} | Split: {} | Recall: {} [BLAS-only]",
                    iter_idx + 1,
                    self.config.iters,
                    self.cost,
                    improvement,
                    self.shift,
                    self.n_split,
                    self.recall,
                );
            } else {
                eprintln!(
                    "Iteration {}/{} | Objective: {} | Improvement: {} | Shift: {} | Split: {} | Recall: {} | Not Pruned %: {} | d': {} -> {}",
                    iter_idx + 1,
                    self.config.iters,
                    self.cost,
                    improvement,
                    self.shift,
                    self.n_split,
                    self.recall,
                    avg_not_pruned * 100.0,
                    old_partial_d,
                    self.partial_d,
                );
            }
        }
    }

    fn should_stop_early(
        &self,
        tracking_recall: bool,
        best_recall: &mut f32,
        iters_without_improvement: &mut usize,
        iter_idx: usize,
    ) -> bool {
        if self.shift < self.config.tol {
            if self.config.verbose {
                eprintln!(
                    "Converged at iteration {} (shift {} < tol {})",
                    iter_idx + 1,
                    self.shift,
                    self.config.tol
                );
            }
            return true;
        }
        if iter_idx > 0 {
            let cost_delta = self.cost / self.prev_cost;
            if cost_delta > 1.0 - self.config.tol {
                if self.config.verbose {
                    eprintln!(
                        "Converged at iteration {} (cost improved by only {})",
                        iter_idx + 1,
                        1.0 - cost_delta
                    );
                }
                return true;
            }
        }
        if tracking_recall {
            let improvement = self.recall - *best_recall;
            if improvement > self.config.recall_tol {
                *best_recall = self.recall;
                *iters_without_improvement = 0;
            } else {
                *iters_without_improvement += 1;
                if *iters_without_improvement >= RECALL_CONVERGENCE_PATIENCE {
                    if self.config.verbose {
                        eprintln!(
                            "Converged at iteration {} (recall {} stalled, best {})",
                            iter_idx + 1,
                            self.recall,
                            best_recall
                        );
                    }
                    return true;
                }
            }
        }
        false
    }

    fn get_output_centroids(&self, should_unrotate: bool) -> Vec<f32> {
        let n_clusters = self.n_clusters;
        let d = self.d;
        if should_unrotate {
            let mut out = vec![0.0; n_clusters * d];
            self.pruner
                .unrotate(&self.horizontal_centroids, &mut out, n_clusters);
            out
        } else {
            self.horizontal_centroids[..n_clusters * d].to_vec()
        }
    }
}

fn get_l2_norms_row_major(data: &[f32], n: usize, d: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= n);
    out[..n]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| {
            let row = &data[i * d..(i + 1) * d];
            let mut sum = 0.0f32;
            for v in row {
                sum += v * v;
            }
            *slot = sum;
        });
}

fn get_partial_l2_norms_row_major(
    data: &[f32],
    n: usize,
    d: usize,
    partial_d: usize,
    out: &mut [f32],
) {
    debug_assert!(out.len() >= n);
    out[..n]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slot)| {
            let row = &data[i * d..i * d + partial_d];
            let mut sum = 0.0f32;
            for v in row {
                sum += v * v;
            }
            *slot = sum;
        });
}

#[allow(dead_code)]
fn _silence_unused_imports(c: KnnCandidate) -> f32 {
    c.distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn make_blobs(n_per_center: usize, n_centers: usize, d: usize, seed: u64) -> (Vec<f32>, usize) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut centers = Vec::with_capacity(n_centers * d);
        for _ in 0..n_centers * d {
            centers.push(rng.gen_range(-10.0..10.0_f32));
        }
        let total = n_per_center * n_centers;
        let mut data = Vec::with_capacity(total * d);
        for c in 0..n_centers {
            for _ in 0..n_per_center {
                for dim in 0..d {
                    data.push(centers[c * d + dim] + rng.gen_range(-0.5..0.5_f32));
                }
            }
        }
        (data, total)
    }

    #[test]
    fn small_blobs_wcss_decreases_with_gemm_only() {
        // d < DIMENSION_THRESHOLD_FOR_PRUNING (128) → always GEMM-only.
        // We can't bound the absolute WCSS without bounding initialisation
        // luck, but Lloyd's algorithm is monotone: each iteration must produce
        // an objective ≤ the previous one (up to FP noise).
        let (data, n) = make_blobs(40, 4, 16, 42);
        let cfg = SuperKMeansConfig {
            iters: 20,
            sampling_fraction: 1.0,
            seed: 7,
            verbose: false,
            // Disable the relative-tolerance early termination so that we get
            // every iteration's stats and can check monotonicity.
            tol: 0.0,
            early_termination: false,
            ..SuperKMeansConfig::default()
        };
        let mut km = SuperKMeans::new(4, 16, cfg);
        let centroids = km.train(&data, n, None, 0);
        assert_eq!(centroids.len(), 4 * 16);
        for s in km.iteration_stats.iter() {
            assert!(s.is_gemm_only, "expected GEMM-only at d=16");
        }
        let stats = &km.iteration_stats;
        for w in stats.windows(2) {
            assert!(
                w[1].objective <= w[0].objective + 1e-3,
                "non-monotone: {} -> {}",
                w[0].objective,
                w[1].objective
            );
        }
    }

    #[test]
    fn small_blobs_converge_well_with_lucky_seed() {
        // Sanity: at least one seed in a small batch should produce a
        // tightly-clustered result (WCSS comparable to within-blob noise).
        let (data, n) = make_blobs(40, 4, 16, 42);
        let mut got_low = false;
        for seed in 0..16u64 {
            let cfg = SuperKMeansConfig {
                iters: 20,
                sampling_fraction: 1.0,
                seed,
                verbose: false,
                ..SuperKMeansConfig::default()
            };
            let mut km = SuperKMeans::new(4, 16, cfg);
            km.train(&data, n, None, 0);
            let final_obj = km.iteration_stats.last().unwrap().objective;
            if final_obj < 500.0 {
                got_low = true;
                break;
            }
        }
        assert!(got_low, "no seed in 0..16 produced WCSS < 500");
    }

    #[test]
    fn high_dim_runs_pruning_path() {
        // Need n_clusters > N_CLUSTERS_THRESHOLD_FOR_PRUNING (256) and
        // d >= DIMENSION_THRESHOLD_FOR_PRUNING (128) to exercise pruning.
        let (data, n) = make_blobs(4, 300, 192, 99);
        let cfg = SuperKMeansConfig {
            iters: 3,
            sampling_fraction: 1.0,
            seed: 11,
            verbose: false,
            ..SuperKMeansConfig::default()
        };
        let mut km = SuperKMeans::new(300, 192, cfg);
        let centroids = km.train(&data, n, None, 0);
        assert_eq!(centroids.len(), 300 * 192);
        assert!(
            km.iteration_stats.iter().any(|s| !s.is_gemm_only),
            "expected at least one pruning iteration"
        );
        // Pruning iterations must report a non-negative not_pruned_pct.
        for s in km.iteration_stats.iter().filter(|s| !s.is_gemm_only) {
            assert!(s.not_pruned_pct >= 0.0);
            assert!(s.partial_d > 0);
        }
    }

    #[test]
    fn pruned_assignments_match_brute_force() {
        // Cross-validate the pruned path against the brute-force `assign()`
        // on the same trained centroids.
        let (data, n) = make_blobs(4, 300, 192, 7);
        let cfg = SuperKMeansConfig {
            iters: 3,
            sampling_fraction: 1.0,
            seed: 11,
            verbose: false,
            unrotate_centroids: false,
            ..SuperKMeansConfig::default()
        };
        let mut km = SuperKMeans::new(300, 192, cfg);
        let centroids = km.train(&data, n, None, 0);
        // Both `assign` and the trained-state assignments are in rotated space
        // (because we kept unrotate_centroids=false). Rotate the inputs to match.
        let mut rotated = vec![0.0; n * 192];
        km.pruner.rotate(&data, &mut rotated, n);
        let brute = km.assign(&rotated, &centroids, n, 300);
        let trained = km.assignments[..n].to_vec();
        let agreed = brute
            .iter()
            .zip(trained.iter())
            .filter(|(a, b)| a == b)
            .count();
        // Allow a small slack: pruned + brute use slightly different distance
        // bookkeeping (squared L2 reconstruction order). We expect ≥ 99% match.
        let agree_pct = agreed as f32 / n as f32;
        assert!(
            agree_pct > 0.99,
            "pruned vs brute force disagreement: {} / {} = {:.4}",
            agreed,
            n,
            agree_pct
        );
    }
}
