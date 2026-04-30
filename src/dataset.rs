use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{
    Array, Float32Array, Float64Array, Int64Array, LargeListArray, ListArray, UInt64Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub struct Dataset {
    pub docs: Vec<Vec<f32>>,
    pub doc_ids: Vec<u64>,
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<u64>>,
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

pub fn load_cohere(
    dataset_dir: &Path,
    n: usize,
    dims: usize,
    num_queries: usize,
    k: usize,
    normalize: bool,
    fetch: bool,
) -> Result<Dataset> {
    if fetch {
        ensure_cohere_dataset(dataset_dir)?;
    }
    let mut docs = read_vector_parquet(
        &dataset_dir.join("shuffle_train.parquet"),
        "id",
        "emb",
        dims,
        n,
    )?;
    let mut queries = read_vector_parquet(
        &dataset_dir.join("test.parquet"),
        "id",
        "emb",
        dims,
        num_queries,
    )?;
    if normalize {
        normalize_all(&mut docs.vectors);
        normalize_all(&mut queries.vectors);
    }
    let ground_truth = read_neighbors_parquet(
        &dataset_dir.join("neighbors.parquet"),
        "neighbors_id",
        num_queries,
        k,
    )?;
    Ok(Dataset {
        docs: docs.vectors,
        doc_ids: docs.ids,
        queries: queries.vectors,
        ground_truth,
    })
}

pub fn top_k_by_score<I>(scores: I, k: usize) -> Vec<(u64, f32)>
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
    let mut top: Vec<_> = heap
        .into_iter()
        .map(|Reverse(scored)| (scored.id, scored.score))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    top
}

pub fn metrics_for(ground_truth_ids: &[u64], got_ranked: &[(u64, f32)], k: usize) -> (f32, f32) {
    let gt: Vec<u64> = ground_truth_ids.iter().copied().take(k).collect();
    let gt_set: std::collections::HashSet<u64> = gt.iter().copied().collect();
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
    (recall, ndcg)
}

struct VectorRows {
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
}

fn ensure_cohere_dataset(dataset_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dataset_dir)
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    for file in ["shuffle_train.parquet", "test.parquet", "neighbors.parquet"] {
        let local = dataset_dir.join(file);
        if local.exists() {
            continue;
        }
        let url = format!("https://assets.zilliz.com/benchmark/cohere_medium_1m/{file}");
        eprintln!("downloading {url}");
        let mut resp = reqwest::blocking::get(&url)
            .with_context(|| format!("GET {url}"))?
            .error_for_status()
            .with_context(|| format!("GET {url}"))?;
        let mut out =
            File::create(&local).with_context(|| format!("create {}", local.display()))?;
        std::io::copy(&mut resp, &mut out).with_context(|| format!("write {}", local.display()))?;
        out.flush()?;
    }
    Ok(())
}

fn read_vector_parquet(
    path: &Path,
    id_col: &str,
    vector_col: &str,
    dims: usize,
    limit: usize,
) -> Result<VectorRows> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.with_batch_size(8192).build()?;
    let mut ids = Vec::with_capacity(limit);
    let mut vectors = Vec::with_capacity(limit);
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let id_idx = batch
            .schema()
            .index_of(id_col)
            .with_context(|| format!("missing column {id_col}"))?;
        let vec_idx = batch
            .schema()
            .index_of(vector_col)
            .with_context(|| format!("missing column {vector_col}"))?;
        let id_array = batch.column(id_idx);
        let vec_array = batch.column(vec_idx);
        for row in 0..batch.num_rows() {
            if ids.len() >= limit {
                return Ok(VectorRows { ids, vectors });
            }
            ids.push(read_u64_scalar(id_array.as_ref(), row)?);
            let vec = read_f32_list(vec_array.as_ref(), row, dims)?;
            vectors.push(vec);
        }
    }
    if ids.len() != limit {
        bail!(
            "expected {limit} rows from {}, got {}",
            path.display(),
            ids.len()
        );
    }
    Ok(VectorRows { ids, vectors })
}

fn read_neighbors_parquet(
    path: &Path,
    neighbors_col: &str,
    limit: usize,
    k: usize,
) -> Result<Vec<Vec<u64>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.with_batch_size(8192).build()?;
    let mut out = Vec::with_capacity(limit);
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let idx = batch
            .schema()
            .index_of(neighbors_col)
            .with_context(|| format!("missing column {neighbors_col}"))?;
        let array = batch.column(idx);
        for row in 0..batch.num_rows() {
            if out.len() >= limit {
                return Ok(out);
            }
            let mut ids = read_u64_list(array.as_ref(), row)?;
            ids.truncate(k);
            out.push(ids);
        }
    }
    if out.len() != limit {
        bail!(
            "expected {limit} rows from {}, got {}",
            path.display(),
            out.len()
        );
    }
    Ok(out)
}

fn read_u64_scalar(array: &dyn Array, row: usize) -> Result<u64> {
    if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(a.value(row));
    }
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(a.value(row) as u64);
    }
    Err(anyhow!(
        "unsupported id column type: {:?}",
        array.data_type()
    ))
}

fn read_f32_list(array: &dyn Array, row: usize, dims: usize) -> Result<Vec<f32>> {
    let values = if let Some(a) = array.as_any().downcast_ref::<ListArray>() {
        let offsets = a.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        read_f32_values(a.values().as_ref(), start, end)?
    } else if let Some(a) = array.as_any().downcast_ref::<LargeListArray>() {
        let offsets = a.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        read_f32_values(a.values().as_ref(), start, end)?
    } else {
        bail!("unsupported vector column type: {:?}", array.data_type());
    };
    if values.len() != dims {
        bail!("expected dim {dims}, got {}", values.len());
    }
    Ok(values)
}

fn read_f32_values(array: &dyn Array, start: usize, end: usize) -> Result<Vec<f32>> {
    if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok((start..end).map(|i| a.value(i)).collect());
    }
    if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok((start..end).map(|i| a.value(i) as f32).collect());
    }
    Err(anyhow!(
        "unsupported vector value type: {:?}",
        array.data_type()
    ))
}

fn read_u64_list(array: &dyn Array, row: usize) -> Result<Vec<u64>> {
    if let Some(a) = array.as_any().downcast_ref::<ListArray>() {
        let offsets = a.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        return read_u64_values(a.values().as_ref(), start, end);
    }
    if let Some(a) = array.as_any().downcast_ref::<LargeListArray>() {
        let offsets = a.value_offsets();
        let start = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        return read_u64_values(a.values().as_ref(), start, end);
    }
    Err(anyhow!(
        "unsupported neighbors column type: {:?}",
        array.data_type()
    ))
}

fn read_u64_values(array: &dyn Array, start: usize, end: usize) -> Result<Vec<u64>> {
    if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok((start..end).map(|i| a.value(i)).collect());
    }
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok((start..end).map(|i| a.value(i) as u64).collect());
    }
    Err(anyhow!(
        "unsupported neighbors value type: {:?}",
        array.data_type()
    ))
}

fn normalize_all(vectors: &mut [Vec<f32>]) {
    for v in vectors {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in v {
            *x /= norm;
        }
    }
}

pub fn default_cohere_dir() -> PathBuf {
    PathBuf::from("/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m")
}
