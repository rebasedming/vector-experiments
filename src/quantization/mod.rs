pub mod experiment;
pub mod factory;
pub mod math;
pub mod naivesq;
pub mod pdx_u8;
pub mod rabitq;
pub mod rotation;
pub mod turboquant;

pub use turboquant::quantizer::QjlProjectionKind;

use std::borrow::Cow;

use anyhow::Result;

use crate::metrics::top_k_by_score;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    L2,
    InnerProduct,
}

#[derive(Debug)]
pub enum VectorError {
    InvalidPersistence(&'static str),
    Io(std::io::Error),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorError::InvalidPersistence(msg) => write!(f, "invalid persisted data: {msg}"),
            VectorError::Io(err) => write!(f, "i/o error: {err}"),
        }
    }
}

impl std::error::Error for VectorError {}

impl From<std::io::Error> for VectorError {
    fn from(err: std::io::Error) -> Self {
        VectorError::Io(err)
    }
}

pub trait VectorQuantizer {
    fn name(&self) -> &'static str;
    fn variant(&self) -> Cow<'_, str> {
        Cow::Borrowed("default")
    }
    fn scoring_layout(&self) -> &'static str {
        "doc-major"
    }
    fn bits(&self) -> u8;
    fn set_transposed(&mut self, _enabled: bool) -> bool {
        false
    }
    fn encode(&mut self, docs: &[Vec<f32>]) -> Result<()>;
    fn score(&self, query: &[f32], doc_idx: usize) -> f32;
    fn top_k(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        top_k_by_score(
            doc_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(doc_idx, doc_id)| (doc_id, self.score(query, doc_idx))),
            k,
        )
    }
    fn bytes_per_vector(&self) -> usize;
    fn total_bytes(&self) -> usize {
        0
    }
}

pub fn compression_ratio(dims: usize, bytes_per_vector: usize) -> f32 {
    (dims * std::mem::size_of::<f32>()) as f32 / bytes_per_vector as f32
}
