use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub(crate) struct ExperimentConfig {
    pub(crate) n: usize,
    pub(crate) dims: usize,
    pub(crate) num_queries: usize,
    pub(crate) k: usize,
    pub(crate) bit_width: u8,
    pub(crate) seed: u64,
    pub(crate) normalize: bool,
    pub(crate) dataset: ExperimentDatasetKind,
}

#[derive(Clone, Debug)]
pub(crate) enum ExperimentDatasetKind {
    Cohere { dir: PathBuf },
    Synthetic,
}

#[derive(Clone, Debug)]
pub(crate) struct ExperimentData {
    pub(crate) docs: Vec<Vec<f32>>,
    pub(crate) doc_ids: Vec<u64>,
    pub(crate) queries: Vec<Vec<f32>>,
    pub(crate) ground_truth: Option<Vec<Vec<u64>>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExperimentMetrics {
    pub(crate) recall: f32,
    pub(crate) ndcg: f32,
}

impl ExperimentConfig {
    pub(crate) fn from_env() -> Self {
        let default_cohere_dir =
            PathBuf::from("/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m");
        let dataset = match std::env::var("TQ_DATASET").ok().as_deref() {
            Some("synthetic") => ExperimentDatasetKind::Synthetic,
            Some("cohere") | None if default_cohere_dir.exists() => {
                let dir = env_path("TQ_COHERE_DIR").unwrap_or(default_cohere_dir);
                ExperimentDatasetKind::Cohere { dir }
            }
            Some("cohere") => {
                let dir = env_path("TQ_COHERE_DIR").unwrap_or(default_cohere_dir);
                ExperimentDatasetKind::Cohere { dir }
            }
            _ => ExperimentDatasetKind::Synthetic,
        };

        let default_n = match dataset {
            ExperimentDatasetKind::Cohere { .. } => 1_000_000,
            ExperimentDatasetKind::Synthetic => 10_000,
        };
        Self {
            n: env_usize("TQ_N", default_n),
            dims: env_usize("TQ_DIMS", 768),
            num_queries: env_usize("TQ_QUERIES", 100),
            k: env_usize("TQ_K", 10),
            bit_width: env_u8("TQ_BITS", 4),
            seed: env_u64("TQ_SEED", 42),
            normalize: env_bool("TQ_NORMALIZE", true),
            dataset,
        }
    }

    pub(crate) fn for_index_from_env() -> Self {
        let mut cfg = Self::from_env();
        cfg.n = env_usize("TQ_INDEX_N", cfg.n.min(100_000));
        cfg.num_queries = env_usize("TQ_INDEX_QUERIES", cfg.num_queries.min(10));
        cfg
    }
}

pub(crate) fn load_experiment_data(cfg: &ExperimentConfig) -> ExperimentData {
    match &cfg.dataset {
        ExperimentDatasetKind::Cohere { dir } => load_cohere_data(cfg, dir)
            .unwrap_or_else(|err| panic!("failed to load Cohere VectorDBBench data: {err}")),
        ExperimentDatasetKind::Synthetic => load_synthetic_data(cfg),
    }
}

pub(crate) fn exact_top_k(
    docs: &[Vec<f32>],
    doc_ids: &[u64],
    query: &[f32],
    k: usize,
) -> Vec<(u64, f32)> {
    top_k_by_score(
        docs.iter()
            .zip(doc_ids.iter().copied())
            .map(|(doc, id)| (id, exact_ip(doc, query))),
        k,
    )
}

pub(crate) fn metrics_for(
    ground_truth_ids: &[u64],
    got_ranked: &[(u64, f32)],
    k: usize,
) -> ExperimentMetrics {
    let gt: Vec<u64> = ground_truth_ids.iter().copied().take(k).collect();
    let gt_set: HashSet<u64> = gt.iter().copied().collect();
    let hits = got_ranked
        .iter()
        .take(k)
        .filter(|(id, _)| gt_set.contains(id))
        .count();
    let recall = hits as f32 / k as f32;

    let dcg = got_ranked
        .iter()
        .take(k)
        .enumerate()
        .filter_map(|(rank, (id, _))| {
            gt_set
                .contains(id)
                .then_some(1.0f32 / ((rank + 2) as f32).log2())
        })
        .sum::<f32>();
    let ideal = (0..gt.len())
        .map(|rank| 1.0f32 / ((rank + 2) as f32).log2())
        .sum::<f32>();
    let ndcg = if ideal > 0.0 { dcg / ideal } else { 0.0 };

    ExperimentMetrics { recall, ndcg }
}

pub(crate) fn top_k_by_score<I>(scores: I, k: usize) -> Vec<(u64, f32)>
where
    I: IntoIterator<Item = (u64, f32)>,
{
    let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(k + 1);
    for (id, score) in scores {
        let candidate = Reverse(ScoredDoc { score, id });
        if heap.len() < k {
            heap.push(candidate);
        } else if let Some(worst) = heap.peek() {
            if candidate.0 > worst.0 {
                heap.pop();
                heap.push(candidate);
            }
        }
    }
    let mut top: Vec<(u64, f32)> = heap
        .into_iter()
        .map(|Reverse(scored)| (scored.id, scored.score))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    top
}

pub(crate) fn print_metric_summary(
    name: &str,
    cfg: &ExperimentConfig,
    metrics: &[ExperimentMetrics],
) {
    let recall = metrics.iter().map(|m| m.recall).sum::<f32>() / metrics.len().max(1) as f32;
    let ndcg = metrics.iter().map(|m| m.ndcg).sum::<f32>() / metrics.len().max(1) as f32;
    eprintln!(
        "{name}: dataset={:?} n={} dims={} queries={} k={} bits={} normalize={} recall@{}={:.4} ndcg@{}={:.4}",
        cfg.dataset,
        cfg.n,
        cfg.dims,
        cfg.num_queries,
        cfg.k,
        cfg.bit_width,
        cfg.normalize,
        cfg.k,
        recall,
        cfg.k,
        ndcg,
    );
}

fn load_synthetic_data(cfg: &ExperimentConfig) -> ExperimentData {
    let docs: Vec<Vec<f32>> = (0..cfg.n)
        .map(|i| unit_rand(cfg.dims, cfg.seed.wrapping_add(1_000 + i as u64)))
        .collect();
    let queries: Vec<Vec<f32>> = (0..cfg.num_queries)
        .map(|i| unit_rand(cfg.dims, cfg.seed.wrapping_add(7_000 + i as u64)))
        .collect();
    let doc_ids: Vec<u64> = (0..cfg.n as u64).collect();
    ExperimentData {
        docs,
        doc_ids,
        queries,
        ground_truth: None,
    }
}

fn load_cohere_data(cfg: &ExperimentConfig, dir: &Path) -> Result<ExperimentData, String> {
    let train = materialize_parquet_vectors(
        dir,
        "shuffle_train.parquet",
        cfg.dims,
        cfg.n,
        "id",
        "emb",
        "train",
    )?;
    let query_limit = cfg.num_queries;
    let test = materialize_parquet_vectors(
        dir,
        "test.parquet",
        cfg.dims,
        query_limit,
        "id",
        "emb",
        "test",
    )?;

    let mut docs = read_f32_matrix(&train.vec_path, cfg.dims)?;
    let mut queries = read_f32_matrix(&test.vec_path, cfg.dims)?;
    if cfg.normalize {
        normalize_all(&mut docs);
        normalize_all(&mut queries);
    }

    let doc_ids = read_u64_vec(&train.id_path)?;
    let ground_truth = if cfg.n == 1_000_000 {
        Some(load_ground_truth(dir, query_limit, cfg.k)?)
    } else {
        None
    };

    Ok(ExperimentData {
        docs,
        doc_ids,
        queries,
        ground_truth,
    })
}

struct MaterializedVectors {
    vec_path: PathBuf,
    id_path: PathBuf,
}

fn materialize_parquet_vectors(
    dir: &Path,
    file_name: &str,
    dims: usize,
    limit: usize,
    id_col: &str,
    vector_col: &str,
    label: &str,
) -> Result<MaterializedVectors, String> {
    let cache_dir =
        env_path("TQ_CACHE_DIR").unwrap_or_else(|| std::env::temp_dir().join("tantivy_tq_cache"));
    std::fs::create_dir_all(&cache_dir).map_err(|e| format!("create cache dir: {e}"))?;
    let stem = format!("{label}_{limit}x{dims}");
    let vec_path = cache_dir.join(format!("{stem}.f32le"));
    let id_path = cache_dir.join(format!("{stem}.u64le"));
    let expected_vec_bytes = limit * dims * std::mem::size_of::<f32>();
    let expected_id_bytes = limit * std::mem::size_of::<u64>();
    if file_has_len(&vec_path, expected_vec_bytes) && file_has_len(&id_path, expected_id_bytes) {
        return Ok(MaterializedVectors { vec_path, id_path });
    }

    let parquet_path = dir.join(file_name);
    if !parquet_path.exists() {
        return Err(format!("missing parquet file {}", parquet_path.display()));
    }

    let python = env_path("TQ_PYTHON")
        .unwrap_or_else(|| PathBuf::from("/Users/mingying/VectorDBBench/.venv/bin/python"));
    let python = if python.exists() {
        python
    } else {
        PathBuf::from("python3")
    };
    let script = r#"
import struct
import sys

import polars as pl

path, id_col, vec_col, dims_s, limit_s, vec_out, id_out = sys.argv[1:]
dims = int(dims_s)
limit = int(limit_s)
df = pl.scan_parquet(path).select([id_col, vec_col]).limit(limit).collect()
if df.height != limit:
    raise RuntimeError(f"expected {limit} rows, got {df.height}")
with open(vec_out, "wb") as vf, open(id_out, "wb") as idf:
    for row_id, vec in zip(df[id_col].to_list(), df[vec_col].to_list()):
        if len(vec) != dims:
            raise RuntimeError(f"expected dim {dims}, got {len(vec)}")
        idf.write(struct.pack("<Q", int(row_id)))
        vf.write(struct.pack("<" + "f" * dims, *[float(x) for x in vec]))
"#;
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(&parquet_path)
        .arg(id_col)
        .arg(vector_col)
        .arg(dims.to_string())
        .arg(limit.to_string())
        .arg(&vec_path)
        .arg(&id_path)
        .output()
        .map_err(|e| format!("run {}: {e}", python.display()))?;
    if !output.status.success() {
        return Err(format!(
            "materialize {file_name} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(MaterializedVectors { vec_path, id_path })
}

fn load_ground_truth(dir: &Path, query_limit: usize, k: usize) -> Result<Vec<Vec<u64>>, String> {
    let cache_dir =
        env_path("TQ_CACHE_DIR").unwrap_or_else(|| std::env::temp_dir().join("tantivy_tq_cache"));
    std::fs::create_dir_all(&cache_dir).map_err(|e| format!("create cache dir: {e}"))?;
    let gt_path = cache_dir.join(format!("neighbors_{query_limit}x{k}.u64le"));
    let expected_bytes = query_limit * k * std::mem::size_of::<u64>();
    if !file_has_len(&gt_path, expected_bytes) {
        let parquet_path = dir.join("neighbors.parquet");
        let python = env_path("TQ_PYTHON")
            .unwrap_or_else(|| PathBuf::from("/Users/mingying/VectorDBBench/.venv/bin/python"));
        let python = if python.exists() {
            python
        } else {
            PathBuf::from("python3")
        };
        let script = r#"
import struct
import sys

import polars as pl

path, limit_s, k_s, out = sys.argv[1:]
limit = int(limit_s)
k = int(k_s)
df = pl.scan_parquet(path).select("neighbors_id").limit(limit).collect()
if df.height != limit:
    raise RuntimeError(f"expected {limit} rows, got {df.height}")
with open(out, "wb") as f:
    for neighbors in df["neighbors_id"].to_list():
        if len(neighbors) < k:
            raise RuntimeError(f"expected at least {k} neighbors, got {len(neighbors)}")
        for doc_id in neighbors[:k]:
            f.write(struct.pack("<Q", int(doc_id)))
"#;
        let output = Command::new(&python)
            .arg("-c")
            .arg(script)
            .arg(&parquet_path)
            .arg(query_limit.to_string())
            .arg(k.to_string())
            .arg(&gt_path)
            .output()
            .map_err(|e| format!("run {}: {e}", python.display()))?;
        if !output.status.success() {
            return Err(format!(
                "materialize neighbors.parquet failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }

    let ids = read_u64_vec(&gt_path)?;
    Ok(ids.chunks_exact(k).map(|chunk| chunk.to_vec()).collect())
}

fn read_f32_matrix(path: &Path, dims: usize) -> Result<Vec<Vec<f32>>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let row_bytes = dims * std::mem::size_of::<f32>();
    if bytes.len() % row_bytes != 0 {
        return Err(format!(
            "{} has {} bytes, not a multiple of row size {row_bytes}",
            path.display(),
            bytes.len()
        ));
    }
    let mut rows = Vec::with_capacity(bytes.len() / row_bytes);
    for row in bytes.chunks_exact(row_bytes) {
        let mut values = Vec::with_capacity(dims);
        for chunk in row.chunks_exact(4) {
            values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        rows.push(values);
    }
    Ok(rows)
}

fn read_u64_vec(path: &Path) -> Result<Vec<u64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() % 8 != 0 {
        return Err(format!(
            "{} has non-u64 byte length {}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn exact_ip(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalize_all(vectors: &mut [Vec<f32>]) {
    for v in vectors {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in v {
            *x /= norm;
        }
    }
}

fn unit_rand(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = Normal::new(0.0_f32, 1.0).unwrap();
    let mut v: Vec<f32> = (0..d).map(|_| n.sample(&mut rng)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn file_has_len(path: &Path, len: usize) -> bool {
    std::fs::metadata(path)
        .map(|meta| meta.len() == len as u64)
        .unwrap_or(false)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u8(name: &str, default: u8) -> u8 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| !matches!(v.as_str(), "0" | "false" | "False" | "FALSE"))
        .unwrap_or(default)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScoredDoc {
    score: f32,
    id: u64,
}

impl Eq for ScoredDoc {}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}
