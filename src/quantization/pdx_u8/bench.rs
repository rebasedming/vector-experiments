use std::borrow::Cow;

use anyhow::Result;

use crate::metrics::top_k_by_score;
use crate::quantization::VectorQuantizer;

use super::search_pipeline::{FlatPdxU8SearchIndex, SearchScratch};

/// PDX global U8 scalar quant + PDX cluster layout + PDXearch approximate search (no IVF).
pub struct PdxU8Bench {
    dims: usize,
    chunk_size: usize,
    seed: u64,
    index: Option<FlatPdxU8SearchIndex>,
}

impl PdxU8Bench {
    pub fn new(dims: usize, chunk_size: usize, seed: u64) -> Self {
        Self {
            dims,
            chunk_size,
            seed,
            index: None,
        }
    }
}

impl VectorQuantizer for PdxU8Bench {
    fn name(&self) -> &'static str {
        "pdx-u8-sq"
    }

    fn variant(&self) -> Cow<'_, str> {
        Cow::Owned(format!(
            "global-affine+pdx-search-chunk{}-seed{}",
            self.chunk_size, self.seed
        ))
    }

    fn scoring_layout(&self) -> &'static str {
        "pdx-cluster-u8"
    }

    fn bits(&self) -> u8 {
        8
    }

    fn set_transposed(&mut self, _enabled: bool) -> bool {
        false
    }

    fn encode(&mut self, docs: &[Vec<f32>]) -> Result<()> {
        anyhow::ensure!(!docs.is_empty(), "docs must not be empty");
        anyhow::ensure!(
            docs[0].len() == self.dims,
            "dimension mismatch: expected {}, got {}",
            self.dims,
            docs[0].len()
        );
        self.index = Some(FlatPdxU8SearchIndex::encode_chunked(
            docs,
            self.chunk_size,
            self.seed,
        )?);
        Ok(())
    }

    fn score(&self, _query: &[f32], _doc_idx: usize) -> f32 {
        f32::NAN
    }

    fn top_k(&self, doc_ids: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let index = self
            .index
            .as_ref()
            .expect("PdxU8Bench::top_k called before encode");
        let mut scratch = SearchScratch::default();
        let hits = index.search(query, k, &mut scratch);
        top_k_by_score(
            hits.into_iter().map(|(corpus_idx, dist)| {
                let doc_id = doc_ids.get(corpus_idx).copied().unwrap_or_else(|| {
                    panic!(
                        "corpus_idx {corpus_idx} out of range for doc_ids (len {})",
                        doc_ids.len()
                    )
                });
                (doc_id, -dist)
            }),
            k,
        )
    }

    fn bytes_per_vector(&self) -> usize {
        self.index
            .as_ref()
            .map(|i| i.dims as usize)
            .unwrap_or(self.dims)
    }

    fn total_bytes(&self) -> usize {
        self.index
            .as_ref()
            .map(|i| i.total_storage_bytes())
            .unwrap_or(0)
    }
}
