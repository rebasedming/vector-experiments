use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{
    Array, Float32Array, Float64Array, Int64Array, LargeListArray, ListArray, UInt64Array,
};
use clap::ValueEnum;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub const COHERE_DIMS: usize = 768;
pub const OPENAI_DIMS: usize = 1536;
pub const BIOASQ_DIMS: usize = 1024;

pub struct Dataset {
    pub name: &'static str,
    pub dims: usize,
    pub docs: Vec<Vec<f32>>,
    pub doc_ids: Vec<u64>,
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<u64>>,
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetSpec {
    pub name: &'static str,
    pub source_name: &'static str,
    pub dir_name: &'static str,
    pub dims: usize,
}

impl DatasetSpec {
    pub fn data_dir(self, dataset_root: &Path) -> PathBuf {
        dataset_root.join(self.source_name).join(self.dir_name)
    }
}

pub const COHERE_MEDIUM: DatasetSpec = DatasetSpec {
    name: "cohere-medium",
    source_name: "cohere",
    dir_name: "cohere_medium_1m",
    dims: COHERE_DIMS,
};

pub const OPENAI_MEDIUM: DatasetSpec = DatasetSpec {
    name: "openai-medium",
    source_name: "openai",
    dir_name: "openai_medium_500k",
    dims: OPENAI_DIMS,
};

pub const BIOASQ_MEDIUM: DatasetSpec = DatasetSpec {
    name: "bioasq-medium",
    source_name: "bioasq",
    dir_name: "bioasq_medium_1m",
    dims: BIOASQ_DIMS,
};

pub const DEFAULT_DATASETS: [DatasetSpec; 3] = [COHERE_MEDIUM, OPENAI_MEDIUM, BIOASQ_MEDIUM];

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DatasetArg {
    /// Run all built-in datasets: Cohere medium, OpenAI medium, and BioASQ medium.
    All,
    /// 1M vectors, 768 dimensions. Wikipedia text embedded with Cohere V2.
    CohereMedium,
    /// 500K vectors, 1536 dimensions. C4 web-crawl text embedded with OpenAI.
    OpenaiMedium,
    /// 1M vectors, 1024 dimensions. BioASQ biomedical text embedded with Cohere V3.
    BioasqMedium,
}

impl DatasetArg {
    pub fn specs(self) -> Vec<DatasetSpec> {
        match self {
            DatasetArg::All => DEFAULT_DATASETS.to_vec(),
            DatasetArg::CohereMedium => vec![COHERE_MEDIUM],
            DatasetArg::OpenaiMedium => vec![OPENAI_MEDIUM],
            DatasetArg::BioasqMedium => vec![BIOASQ_MEDIUM],
        }
    }
}

pub fn load_dataset(
    spec: DatasetSpec,
    dataset_dir: &Path,
    doc_limit: Option<usize>,
    fetch: bool,
) -> Result<Dataset> {
    if fetch {
        ensure_dataset(spec, dataset_dir)?;
    }
    let mut docs = read_vector_parquet(
        &dataset_dir.join("shuffle_train.parquet"),
        "id",
        "emb",
        spec.dims,
        doc_limit,
    )?;
    let mut queries = read_vector_parquet(
        &dataset_dir.join("test.parquet"),
        "id",
        "emb",
        spec.dims,
        None,
    )?;
    normalize_all(&mut docs.vectors);
    normalize_all(&mut queries.vectors);
    let ground_truth =
        read_neighbors_parquet(&dataset_dir.join("neighbors.parquet"), "neighbors_id")?;
    Ok(Dataset {
        name: spec.name,
        dims: spec.dims,
        docs: docs.vectors,
        doc_ids: docs.ids,
        queries: queries.vectors,
        ground_truth,
    })
}

struct VectorRows {
    ids: Vec<u64>,
    vectors: Vec<Vec<f32>>,
}

fn ensure_dataset(spec: DatasetSpec, dataset_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dataset_dir)
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    for file in ["shuffle_train.parquet", "test.parquet", "neighbors.parquet"] {
        let local = dataset_dir.join(file);
        if local.exists() {
            continue;
        }
        let url = format!(
            "https://assets.zilliz.com/benchmark/{}/{file}",
            spec.dir_name
        );
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
    limit: Option<usize>,
) -> Result<VectorRows> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.with_batch_size(8192).build()?;
    let mut ids = Vec::with_capacity(limit.unwrap_or(0));
    let mut vectors = Vec::with_capacity(limit.unwrap_or(0));
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
            if limit.is_some_and(|limit| ids.len() >= limit) {
                return Ok(VectorRows { ids, vectors });
            }
            ids.push(read_u64_scalar(id_array.as_ref(), row)?);
            let vec = read_f32_list(vec_array.as_ref(), row, dims)?;
            vectors.push(vec);
        }
    }
    if let Some(limit) = limit {
        if ids.len() != limit {
            bail!(
                "expected {limit} rows from {}, got {}",
                path.display(),
                ids.len()
            );
        }
    }
    Ok(VectorRows { ids, vectors })
}

fn read_neighbors_parquet(path: &Path, neighbors_col: &str) -> Result<Vec<Vec<u64>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.with_batch_size(8192).build()?;
    let mut out = Vec::new();
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let idx = batch
            .schema()
            .index_of(neighbors_col)
            .with_context(|| format!("missing column {neighbors_col}"))?;
        let array = batch.column(idx);
        for row in 0..batch.num_rows() {
            out.push(read_u64_list(array.as_ref(), row)?);
        }
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

pub fn default_dataset_root() -> PathBuf {
    PathBuf::from("/tmp/vectordb_bench/dataset")
}
