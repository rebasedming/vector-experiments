use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use crate::dataset::Dataset;
use crate::experiment::{Experiment, ExperimentOutput};
use crate::kmeans::distance::{find_k_nearest_neighbors, X_BATCH_SIZE, Y_BATCH_SIZE};
use crate::kmeans::spann::{
    duplicate_assignments, duplicate_assignments_via_tree, duplicated_cluster_sizes,
    hierarchical_balanced_kmeans, BisectTree,
};
use crate::kmeans::vanilla;

#[derive(Debug, Clone, Args)]
pub struct KMeansExperiment {
    /// Centroids as a fraction of corpus size. The cluster count `k` is
    /// computed at run time as `round(n * centroid_ratio)` for each dataset.
    /// Comma-separated to sweep multiple ratios in one run; SPANN reports
    /// saturation around 0.16 (16% of n).
    #[arg(long, value_delimiter = ',', default_values_t = vec![0.001f32])]
    pub centroid_ratio: Vec<f32>,

    /// Maximum k-means iterations (Lloyd's path only; ignored if --hierarchical).
    #[arg(long, default_value_t = 10)]
    pub iters: u32,

    /// Sampling fraction (0.0, 1.0]. (Lloyd's path only.)
    #[arg(long, default_value_t = 0.3)]
    pub sampling_fraction: f32,

    /// Maximum points per cluster (FAISS-style sampling cap).
    #[arg(long, default_value_t = 256)]
    pub max_points_per_cluster: u32,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Print per-iteration progress.
    #[arg(long, default_value_t = false)]
    pub verbose: bool,

    /// SPANN: use hierarchical balanced bisection instead of flat Lloyd's.
    /// Each level recursively halves the dataset with an exact size split,
    /// producing leaf clusters of approximately `n/k` points each.
    #[arg(long, default_value_t = false)]
    pub hierarchical: bool,

    /// SPANN: balance-Lloyd refinement iterations per bisection step.
    #[arg(long, default_value_t = 5)]
    pub hierarchical_iters_per_bisect: usize,

    /// SPANN: maximum cluster fan-out per point (≤1 disables duplication).
    /// Boundary points within `--duplicate-ratio` of the primary's distance
    /// get added to up to this many additional clusters.
    #[arg(long, default_value_t = 1)]
    pub duplicate_max: usize,

    /// SPANN: distance ratio threshold for boundary duplication. A runner-up
    /// centroid duplicates the point only if its distance ≤ ratio × primary
    /// (applied to plain L2 distance, i.e. squared L2 ≤ ratio² × primary²).
    #[arg(long, default_value_t = 1.0)]
    pub duplicate_ratio: f32,

    /// (tantivy variant only) Number of independent restarts; the run with
    /// the lowest WCSS objective is kept.
    #[arg(long, default_value_t = 1)]
    pub nredo: usize,

    /// (tantivy variant only) Spherical k-means: normalize centroids to unit
    /// length each iteration. Useful for cosine-style retrieval.
    #[arg(long, default_value_t = false)]
    pub spherical: bool,

    /// Recall@k values to evaluate. Comma-separated; e.g. `1,10,100`.
    #[arg(long, value_delimiter = ',', default_values_t = vec![1usize, 10, 100])]
    pub recall_k: Vec<usize>,

    /// Cluster-fanout fractions (top-N centroids visited, as a fraction of
    /// total clusters) to evaluate. Comma-separated; e.g. `0.001,0.01,0.05`.
    #[arg(long, value_delimiter = ',', default_values_t = vec![0.001f32, 0.005, 0.01, 0.05, 0.1])]
    pub recall_n: Vec<f32>,
}

impl Experiment for KMeansExperiment {
    fn name(&self) -> &'static str {
        "kmeans"
    }

    fn run(&self, data: &Dataset) -> Result<ExperimentOutput> {
        let dims = data.dims;
        let n = data.docs.len();
        let mut flat = Vec::with_capacity(n * dims);
        for v in &data.docs {
            flat.extend_from_slice(v);
        }

        let mut header: Vec<String> = vec![
            "variant".to_string(),
            "ratio".to_string(),
            "k".to_string(),
            "total_ms".to_string(),
            "fanout".to_string(),
        ];
        for k_val in &self.recall_k {
            header.push(format!("r@{}", k_val));
        }
        header.extend([
            "min_size".to_string(),
            "median_size".to_string(),
            "max_size".to_string(),
            "memory".to_string(),
            "disk".to_string(),
        ]);
        let mut output = ExperimentOutput::new(header);

        for &ratio in &self.centroid_ratio {
            let k = ((n as f32 * ratio).round() as usize).clamp(1, n);
            self.run_one_ratio(data, &flat, n, dims, ratio, k, &mut output);
        }

        Ok(output)
    }
}

impl KMeansExperiment {
    #[allow(clippy::too_many_arguments)]
    fn run_one_ratio(
        &self,
        data: &Dataset,
        flat: &[f32],
        n: usize,
        dims: usize,
        ratio: f32,
        k: usize,
        output: &mut ExperimentOutput,
    ) {
        // -------- 1. Train: produce centroids + primary assignments --------
        let mut train_ms = 0.0f64;
        let mut bisect_tree: Option<BisectTree> = None;
        let (variant_label, centroids, primary_assignments): (&str, Vec<f32>, Vec<u32>) =
            if self.hierarchical {
                let t = Instant::now();
                let (c, a, tree) = hierarchical_balanced_kmeans(
                    flat,
                    n,
                    dims,
                    k,
                    self.hierarchical_iters_per_bisect,
                    self.seed,
                );
                train_ms = t.elapsed().as_secs_f64() * 1000.0;
                bisect_tree = Some(tree);
                ("hierarchical", c, a)
            } else {
                let (c, a) =
                    self.run_vanilla(&data.docs, n, dims, k, &mut train_ms);
                ("vanilla", c, a)
            };

        eprintln!(
            "  [ratio={:.3} k={}] train={:.1}s",
            ratio,
            k,
            train_ms / 1000.0,
        );

        // -------- 2. Boundary duplication (optional) --------
        let dup_enabled = self.duplicate_max > 1 && self.duplicate_ratio > 1.0;
        let dup_t = Instant::now();
        let multi_assignments: Vec<Vec<u32>> = if dup_enabled {
            if let Some(tree) = &bisect_tree {
                duplicate_assignments_via_tree(
                    flat,
                    n,
                    dims,
                    &centroids,
                    k,
                    &primary_assignments,
                    self.duplicate_max,
                    self.duplicate_ratio,
                    tree,
                    4, // budget = max_copies × 4 leaves visited per query
                )
            } else {
                duplicate_assignments(
                    flat,
                    n,
                    dims,
                    &centroids,
                    k,
                    &primary_assignments,
                    self.duplicate_max,
                    self.duplicate_ratio,
                )
            }
        } else {
            primary_assignments.iter().map(|&c| vec![c]).collect()
        };
        let dup_ms = dup_t.elapsed().as_secs_f64() * 1000.0;
        if dup_enabled {
            eprintln!("  [ratio={:.3} k={}] duplicate={:.1}s", ratio, k, dup_ms / 1000.0);
        }

        // -------- 3. Cluster size distribution + storage --------
        let cluster_sizes: Vec<u32> = if dup_enabled {
            duplicated_cluster_sizes(&multi_assignments, k)
        } else {
            let mut sizes = vec![0u32; k];
            for &c in &primary_assignments {
                sizes[c as usize] += 1;
            }
            sizes
        };
        let (size_min, size_median, size_max) = min_median_max(&cluster_sizes);

        let bytes_per_vec = dims * 4;
        let total_doc_copies: u64 = cluster_sizes.iter().map(|&s| s as u64).sum();
        let centroid_bytes = (k as u64) * bytes_per_vec as u64;
        let posting_bytes = total_doc_copies * bytes_per_vec as u64;
        let mem_str = format_bytes(centroid_bytes);
        let disk_str = format_bytes(posting_bytes);

        // -------- 4. Recall --------
        let recall_t = Instant::now();
        let recall = compute_recall(
            data,
            dims,
            k,
            &centroids,
            &multi_assignments,
            &self.recall_k,
            &self.recall_n,
        );
        let recall_ms = recall_t.elapsed().as_secs_f64() * 1000.0;
        if recall.is_some() {
            eprintln!("  [ratio={:.3} k={}] recall={:.1}s", ratio, k, recall_ms / 1000.0);
        }

        // -------- 5. Emit rows --------
        let total_str = format!("{:.1}", train_ms);
        let ratio_str = format!("{:.3}", ratio);
        let k_str = k.to_string();

        if let Some(recall) = &recall {
            for (n_idx, n_frac) in self.recall_n.iter().enumerate() {
                let n_count = recall.n_counts[n_idx];
                let mut row = vec![
                    variant_label.to_string(),
                    ratio_str.clone(),
                    k_str.clone(),
                    total_str.clone(),
                    format!("{:.2}% (n={})", n_frac * 100.0, n_count),
                ];
                for k_idx in 0..self.recall_k.len() {
                    row.push(format!("{:.4}", recall.values[k_idx][n_idx]));
                }
                row.extend([
                    size_min.to_string(),
                    format!("{:.1}", size_median),
                    size_max.to_string(),
                    mem_str.clone(),
                    disk_str.clone(),
                ]);
                output.push_row(row);
            }
        } else {
            let mut row = vec![
                variant_label.to_string(),
                ratio_str,
                k_str,
                total_str,
                "-".to_string(),
            ];
            for _ in &self.recall_k {
                row.push("-".to_string());
            }
            row.extend([
                size_min.to_string(),
                format!("{:.1}", size_median),
                size_max.to_string(),
                mem_str,
                disk_str,
            ]);
            output.push_row(row);
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn min_median_max(sizes: &[u32]) -> (u32, f32, u32) {
    if sizes.is_empty() {
        return (0, 0.0, 0);
    }
    let mut sorted = sizes.to_vec();
    sorted.sort_unstable();
    let min = *sorted.first().unwrap();
    let max = *sorted.last().unwrap();
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 1 {
        sorted[mid] as f32
    } else {
        (sorted[mid - 1] as f32 + sorted[mid] as f32) / 2.0
    };
    (min, median, max)
}

struct RecallResult {
    n_counts: Vec<usize>,
    /// values[k_idx][n_idx]
    values: Vec<Vec<f32>>,
}

impl KMeansExperiment {
    fn run_vanilla(
        &self,
        docs: &[Vec<f32>],
        n: usize,
        dims: usize,
        k: usize,
        train_ms: &mut f64,
    ) -> (Vec<f32>, Vec<u32>) {
        // dot_products buffer is sized chunk × k × 4 bytes per rayon thread.
        // Cap it at ~64 MB per thread to keep total memory bounded as k grows.
        let target_bytes_per_thread: usize = 64 * 1024 * 1024;
        let decode_block_size =
            (target_bytes_per_thread / (k.max(1) * 4)).clamp(64, 32_768);

        let cfg = vanilla::KMeansConfig {
            niter: self.iters as usize,
            nredo: self.nredo,
            seed: self.seed,
            spherical: self.spherical,
            max_points_per_centroid: self.max_points_per_cluster as usize,
            decode_block_size,
        };
        let t = Instant::now();
        let result = vanilla::run_kmeans_with_config(docs, k, cfg);
        *train_ms = t.elapsed().as_secs_f64() * 1000.0;

        let mut centroids = Vec::with_capacity(k * dims);
        for c in &result.centroids {
            centroids.extend_from_slice(c);
        }
        let assignments: Vec<u32> = result.assignments.iter().map(|&a| a as u32).collect();
        debug_assert_eq!(assignments.len(), n);
        (centroids, assignments)
    }
}

/// For each query, find the closest `max_n` centroids; then for each (k, n)
/// pair report the fraction of GT-top-k corpus neighbours whose **multi-cluster
/// assignment** intersects the query's top-n centroid set.
fn compute_recall(
    data: &Dataset,
    dims: usize,
    n_clusters: usize,
    centroids: &[f32],
    assignments: &[Vec<u32>],
    recall_k: &[usize],
    recall_n_fracs: &[f32],
) -> Option<RecallResult> {
    let n_queries = data.queries.len();
    if n_queries == 0 || data.ground_truth.is_empty() {
        return None;
    }
    let recall_n_counts: Vec<usize> = recall_n_fracs
        .iter()
        .map(|frac| {
            let raw = (frac * n_clusters as f32).round() as i64;
            raw.max(1).min(n_clusters as i64) as usize
        })
        .collect();
    let max_n = recall_n_counts.iter().copied().max().unwrap_or(0);
    if max_n == 0 {
        return None;
    }

    let mut q_flat = Vec::with_capacity(n_queries * dims);
    for v in &data.queries {
        q_flat.extend_from_slice(v);
    }

    let mut q_norms = vec![0.0f32; n_queries];
    for (i, slot) in q_norms.iter_mut().enumerate() {
        let row = &q_flat[i * dims..(i + 1) * dims];
        *slot = row.iter().map(|v| v * v).sum();
    }
    let mut c_norms = vec![0.0f32; n_clusters];
    for (i, slot) in c_norms.iter_mut().enumerate() {
        let row = &centroids[i * dims..(i + 1) * dims];
        *slot = row.iter().map(|v| v * v).sum();
    }

    let mut top_centroids = vec![0u32; n_queries * max_n];
    let mut top_distances = vec![0.0f32; n_queries * max_n];
    let mut tmp = vec![0.0f32; X_BATCH_SIZE * Y_BATCH_SIZE];
    find_k_nearest_neighbors(
        &q_flat,
        centroids,
        n_queries,
        n_clusters,
        dims,
        &q_norms,
        &c_norms,
        max_n,
        &mut top_centroids,
        &mut top_distances,
        &mut tmp,
    );

    let mut doc_id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(data.doc_ids.len());
    for (i, &id) in data.doc_ids.iter().enumerate() {
        doc_id_to_idx.insert(id, i);
    }

    let mut recall = vec![vec![0.0f32; recall_n_counts.len()]; recall_k.len()];

    for q in 0..n_queries {
        let gt = &data.ground_truth[q];
        let q_top = &top_centroids[q * max_n..(q + 1) * max_n];

        let max_k = recall_k.iter().copied().max().unwrap_or(0).min(gt.len());
        // Per GT entry, store the entire cluster list for the matching doc.
        let mut gt_clusters: Vec<&[u32]> = Vec::with_capacity(max_k);
        for &doc_id in gt.iter().take(max_k) {
            if let Some(&corpus_idx) = doc_id_to_idx.get(&doc_id) {
                gt_clusters.push(&assignments[corpus_idx]);
            }
        }
        let gt_len = gt_clusters.len();

        for (n_idx, &n_val) in recall_n_counts.iter().enumerate() {
            let cluster_set: HashSet<u32> = q_top[..n_val].iter().copied().collect();
            for (k_idx, &k_val) in recall_k.iter().enumerate() {
                let k_clamped = k_val.min(gt_len);
                if k_clamped == 0 {
                    continue;
                }
                let hits = gt_clusters[..k_clamped]
                    .iter()
                    .filter(|cs| cs.iter().any(|c| cluster_set.contains(c)))
                    .count();
                recall[k_idx][n_idx] += hits as f32 / k_clamped as f32;
            }
        }
    }
    for row in recall.iter_mut() {
        for v in row.iter_mut() {
            *v /= n_queries as f32;
        }
    }

    Some(RecallResult {
        n_counts: recall_n_counts,
        values: recall,
    })
}
