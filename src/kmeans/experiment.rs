use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use crate::dataset::Dataset;
use crate::experiment::{Experiment, ExperimentOutput};
use crate::kmeans::spann::{
    balance_from_sizes, duplicate_assignments, duplicated_cluster_sizes,
    hierarchical_balanced_kmeans,
};
use crate::kmeans::superkmeans::batch::find_k_nearest_neighbors;
use crate::kmeans::superkmeans::common::{X_BATCH_SIZE, Y_BATCH_SIZE};
use crate::kmeans::superkmeans::{SuperKMeans, SuperKMeansConfig};

#[derive(Debug, Clone, Args)]
pub struct KMeansExperiment {
    /// Number of clusters to train.
    #[arg(long, default_value_t = 1024)]
    pub k: usize,

    /// Maximum k-means iterations (Lloyd's path only; ignored if --hierarchical).
    #[arg(long, default_value_t = 10)]
    pub iters: u32,

    /// Sampling fraction (0.0, 1.0]. (Lloyd's path only.)
    #[arg(long, default_value_t = 0.3)]
    pub sampling_fraction: f32,

    /// Maximum points per cluster (FAISS-style sampling cap).
    #[arg(long, default_value_t = 256)]
    pub max_points_per_cluster: u32,

    /// Enable all SuperKMeans optimisations on top of vanilla Lloyd's:
    /// random orthogonal rotation, ADSampling-based dimension pruning over a
    /// PDX block layout, and early termination on centroid-shift / cost /
    /// recall tolerance. Off by default — without this flag the experiment
    /// runs plain GEMM-based Lloyd's iteration with no pruning, no rotation,
    /// and a fixed `--iters` count.
    #[arg(long, default_value_t = false)]
    pub superkmeans: bool,

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
        "kmeans/superkmeans"
    }

    fn run(&self, data: &Dataset) -> Result<ExperimentOutput> {
        let dims = data.dims;
        let n = data.docs.len();
        let mut flat = Vec::with_capacity(n * dims);
        for v in &data.docs {
            flat.extend_from_slice(v);
        }

        let mut output = ExperimentOutput::new([
            "stage", "objective", "shift", "split", "not_pruned_pct", "partial_d", "gemm_only", "ms",
        ]);

        // -------- 1. Train: produce centroids + primary assignments --------
        let mut setup_ms = 0.0f64;
        let mut train_ms = 0.0f64;
        let mut assign_ms = 0.0f64;
        let (centroids, primary_assignments): (Vec<f32>, Vec<u32>) = if self.hierarchical {
            let t = Instant::now();
            let (c, a) = hierarchical_balanced_kmeans(
                &flat,
                n,
                dims,
                self.k,
                self.hierarchical_iters_per_bisect,
                self.seed,
            );
            train_ms = t.elapsed().as_secs_f64() * 1000.0;
            output.push_row([
                "HIERARCH".to_string(),
                format!("levels={}", (self.k as f32).log2().ceil() as u32),
                format!("iters/bisect={}", self.hierarchical_iters_per_bisect),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ]);
            (c, a)
        } else {
            // Vanilla Lloyd's by default; `--superkmeans` flips on rotation,
            // PDX + ADSampling pruning, and the convergence-based early stop.
            let cfg = SuperKMeansConfig {
                iters: self.iters,
                sampling_fraction: self.sampling_fraction,
                max_points_per_cluster: self.max_points_per_cluster,
                seed: self.seed,
                use_blas_only: !self.superkmeans,
                early_termination: self.superkmeans,
                data_already_rotated: !self.superkmeans,
                verbose: self.verbose,
                ..SuperKMeansConfig::default()
            };
            let t_setup = Instant::now();
            let mut km = SuperKMeans::new(self.k, dims, cfg);
            setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;

            let t_train = Instant::now();
            let centroids = km.train(&flat, n, None, 0);
            train_ms = t_train.elapsed().as_secs_f64() * 1000.0;

            for stat in &km.iteration_stats {
                output.push_row([
                    stat.iteration.to_string(),
                    format!("{:.4}", stat.objective),
                    format!("{:.6}", stat.shift),
                    stat.split.to_string(),
                    format!("{:.4}", stat.not_pruned_pct),
                    stat.partial_d.to_string(),
                    stat.is_gemm_only.to_string(),
                    format!("{:.1}", stat.duration_ms),
                ]);
            }

            // With --superkmeans we use the pruning fast-path
            // (assign_training_points), which on non-AMX hardware can be
            // dramatically faster than brute-force; without the flag we fall
            // back to brute-force `assign` (the pruning ratios assume the
            // rotation has been applied, so they're not valid otherwise).
            let t_assign = Instant::now();
            let primary = if self.superkmeans {
                km.assign_training_points(&flat, &centroids, n, self.k)
            } else {
                km.assign(&flat, &centroids, n, self.k)
            };
            assign_ms = t_assign.elapsed().as_secs_f64() * 1000.0;
            (centroids, primary)
        };

        // -------- 2. Primary balance summary --------
        let primary_balance = SuperKMeans::cluster_balance_stats(&primary_assignments, n, self.k);
        if setup_ms > 0.0 {
            output.push_row([
                "SETUP".to_string(),
                "pruner+buffers".to_string(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                format!("{:.1}", setup_ms),
            ]);
        }
        output.push_row([
            "TRAIN".to_string(),
            format!("n={}", n),
            format!("min={}", primary_balance.min),
            format!("max={}", primary_balance.max),
            format!("mean={:.1}", primary_balance.mean),
            format!("CV={:.3}", primary_balance.cv),
            String::new(),
            format!("{:.1}", train_ms),
        ]);
        if assign_ms > 0.0 {
            let method = if self.superkmeans { "pruned" } else { "brute" };
            output.push_row([
                "ASSIGN".to_string(),
                format!("{}, k={}", method, self.k),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                format!("{:.1}", assign_ms),
            ]);
        }

        // -------- 3. Boundary duplication (optional) --------
        let dup_enabled = self.duplicate_max > 1 && self.duplicate_ratio > 1.0;
        let dup_start = Instant::now();
        let multi_assignments = if dup_enabled {
            duplicate_assignments(
                &flat,
                n,
                dims,
                &centroids,
                self.k,
                &primary_assignments,
                self.duplicate_max,
                self.duplicate_ratio,
            )
        } else {
            primary_assignments.iter().map(|&c| vec![c]).collect()
        };
        let dup_ms = dup_start.elapsed().as_secs_f64() * 1000.0;
        if dup_enabled {
            let total: usize = multi_assignments.iter().map(|v| v.len()).sum();
            let avg_copies = total as f32 / n as f32;
            let dup_sizes = duplicated_cluster_sizes(&multi_assignments, self.k);
            let (mean, _, min, max, cv) = balance_from_sizes(&dup_sizes);
            output.push_row([
                "DUPLICATE".to_string(),
                format!("avg_copies={:.2}", avg_copies),
                format!("min={}", min),
                format!("max={}", max),
                format!("mean={:.1}", mean),
                format!("CV={:.3}", cv),
                String::new(),
                format!("{:.1}", dup_ms),
            ]);
        }

        // -------- 4. Recall --------
        let recall_section = compute_recall_section(
            data,
            dims,
            self.k,
            &centroids,
            &multi_assignments,
            &self.recall_k,
            &self.recall_n,
        );
        if let Some((section, recall_ms)) = recall_section {
            output.push_row([
                "RECALL".to_string(),
                format!("queries={}", data.queries.len()),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                format!("{:.1}", recall_ms),
            ]);
            output.push_extra(section);
        }

        let total_ms = setup_ms + train_ms + assign_ms + dup_ms;
        output.push_row([
            "TOTAL".to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            format!("{:.1}", total_ms),
        ]);
        Ok(output)
    }
}

/// For each query, find the closest `max_n` centroids; then for each (k, n)
/// pair report the fraction of GT-top-k corpus neighbours whose **multi-cluster
/// assignment** intersects the query's top-n centroid set.
fn compute_recall_section(
    data: &Dataset,
    dims: usize,
    n_clusters: usize,
    centroids: &[f32],
    assignments: &[Vec<u32>],
    recall_k: &[usize],
    recall_n_fracs: &[f32],
) -> Option<(String, f64)> {
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

    let recall_start = Instant::now();

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

    let recall_ms = recall_start.elapsed().as_secs_f64() * 1000.0;

    let header: Vec<String> = std::iter::once("recall".to_string())
        .chain(
            recall_n_fracs
                .iter()
                .zip(recall_n_counts.iter())
                .map(|(frac, count)| format!("{:.2}% (n={})", frac * 100.0, count)),
        )
        .collect();
    let rows: Vec<Vec<String>> = recall_k
        .iter()
        .enumerate()
        .map(|(k_idx, k_val)| {
            let mut row = vec![format!("top-{k_val}")];
            for n_idx in 0..recall_n_counts.len() {
                row.push(format!("{:.4}", recall[k_idx][n_idx]));
            }
            row
        })
        .collect();
    let section = render_markdown_table(&header, &rows);
    Some((section, recall_ms))
}

fn render_markdown_table(header: &[String], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = header.iter().map(|c| c.len()).collect();
    for row in rows {
        for (i, c) in row.iter().enumerate() {
            widths[i] = widths[i].max(c.len());
        }
    }
    let numeric: Vec<bool> = (0..header.len())
        .map(|i| rows.iter().all(|r| r[i].parse::<f64>().is_ok()))
        .collect();
    let mut out = String::new();
    push_row(&mut out, header, &widths, &numeric);
    push_separator(&mut out, &widths);
    for row in rows {
        push_row(&mut out, row, &widths, &numeric);
    }
    out
}

fn push_row(out: &mut String, row: &[String], widths: &[usize], numeric: &[bool]) {
    out.push('|');
    for (i, cell) in row.iter().enumerate() {
        if numeric[i] {
            out.push_str(&format!(" {:>w$} |", cell, w = widths[i]));
        } else {
            out.push_str(&format!(" {:<w$} |", cell, w = widths[i]));
        }
    }
    out.push('\n');
}

fn push_separator(out: &mut String, widths: &[usize]) {
    out.push('|');
    for w in widths {
        out.push_str(&"-".repeat(w + 2));
        out.push('|');
    }
    out.push('\n');
}
