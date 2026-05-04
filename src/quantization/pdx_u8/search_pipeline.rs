//! PDXearch `Search()` core (`searcher.hpp`) for **`Quantization::U8`** without IVF centroid ordering:
//! clusters are visited **sequentially**. Scalar distance kernels (`distance_u8.rs`).
//!
//! Chunking (`chunk_size`) is required so Warmup/Prune run once the heap holds `k` candidates.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::adsampling::AdsamplingPruner;
use super::distance_u8::{horizontal_chunk, vertical_full, vertical_pruning, DistanceU32};
use super::layout::{
    cluster_buffer_bytes, get_pdx_dimension_split, normalize_query, quantize_embedding,
    scatter_u8_embedding_into_cluster, ScalarQuantParams, MAX_U8,
};

/// PDX `DIMENSIONS_FETCHING_SIZES` (`common.hpp`).
pub const DIMENSIONS_FETCHING_SIZES: [usize; 20] = [
    16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 512, 1024, 2048, 16384,
];

const MIN_MAX_CAPACITY: usize = 256;

#[derive(Clone)]
struct KnCand {
    dist: f32,
    corpus_idx: u32,
}

impl PartialEq for KnCand {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.corpus_idx == other.corpus_idx
    }
}
impl Eq for KnCand {}

impl PartialOrd for KnCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KnCand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.corpus_idx.cmp(&other.corpus_idx))
    }
}

#[inline]
fn heap_maybe_push(heap: &mut BinaryHeap<KnCand>, k: usize, cand: KnCand) {
    if heap.len() < k {
        heap.push(cand);
        return;
    }
    if let Some(worst) = heap.peek() {
        if cand.dist < worst.dist {
            heap.pop();
            heap.push(cand);
        }
    }
}

pub struct PdxClusterBuffer {
    pub num_embeddings: usize,
    pub max_capacity: usize,
    pub data: Vec<u8>,
    pub corpus_indices: Vec<u32>,
}

pub struct FlatPdxU8SearchIndex {
    pub dims: u32,
    pub vertical_dimensions: u32,
    pub horizontal_dimensions: u32,
    pub params: ScalarQuantParams,
    pub quantization_scale_squared: f32,
    pub inverse_quantization_scale_squared: f32,
    pub clusters: Vec<PdxClusterBuffer>,
    pub pruner: AdsamplingPruner,
    pub selectivity_threshold: f32,
}

impl FlatPdxU8SearchIndex {
    pub fn max_cluster_capacity(&self) -> usize {
        self.clusters
            .iter()
            .map(|c| c.max_capacity)
            .max()
            .unwrap_or(0)
    }

    pub fn encode_chunked(docs: &[Vec<f32>], chunk_size: usize, seed: u64) -> anyhow::Result<Self> {
        anyhow::ensure!(!docs.is_empty(), "docs must not be empty");
        let dims = docs[0].len() as u32;
        for d in docs {
            anyhow::ensure!(d.len() as u32 == dims, "dimension mismatch");
        }
        let split = get_pdx_dimension_split(dims);
        let vd = split.vertical_dimensions;
        let hd = split.horizontal_dimensions;

        // ADSampling compares query vs corpus in the **same rotated basis** (`adsampling.hpp`).
        let pruner = AdsamplingPruner::new(dims as usize, seed);

        let d_usize = dims as usize;
        let mut rot_buf = vec![0f32; d_usize];
        let mut global_min = f32::MAX;
        let mut global_max = f32::NEG_INFINITY;
        for doc in docs {
            pruner.rotate_embedding(doc, &mut rot_buf);
            for &x in rot_buf.iter() {
                global_min = global_min.min(x);
                global_max = global_max.max(x);
            }
        }
        let range = global_max - global_min;
        let quantization_scale = if range > 0.0 {
            (MAX_U8 as f32) / range
        } else {
            1.0
        };
        let params = ScalarQuantParams {
            quantization_base: global_min,
            quantization_scale,
        };
        let quant_scale_sq = params.quantization_scale * params.quantization_scale;
        let inv_qss = 1.0 / quant_scale_sq;

        let chunk_size = chunk_size.max(1);
        let mut clusters = Vec::new();
        let mut corpus_off = 0usize;
        while corpus_off < docs.len() {
            let end = (corpus_off + chunk_size).min(docs.len());
            let n_emb = end - corpus_off;
            let max_capacity = (((n_emb as f32) * 1.3).ceil() as usize)
                .max(MIN_MAX_CAPACITY)
                .max(n_emb);
            let stride = max_capacity;
            let mut data = vec![0u8; cluster_buffer_bytes(stride, dims)];
            let mut corpus_indices = Vec::with_capacity(n_emb);
            let mut code_row = vec![0u8; dims as usize];

            for i in 0..n_emb {
                let doc = &docs[corpus_off + i];
                pruner.rotate_embedding(doc, &mut rot_buf);
                quantize_embedding(&rot_buf, params, &mut code_row);
                scatter_u8_embedding_into_cluster(&mut data, stride, i, &code_row, dims);
                corpus_indices.push((corpus_off + i) as u32);
            }

            clusters.push(PdxClusterBuffer {
                num_embeddings: n_emb,
                max_capacity: stride,
                data,
                corpus_indices,
            });
            corpus_off = end;
        }

        Ok(Self {
            dims,
            vertical_dimensions: vd,
            horizontal_dimensions: hd,
            params,
            quantization_scale_squared: quant_scale_sq,
            inverse_quantization_scale_squared: inv_qss,
            clusters,
            pruner,
            selectivity_threshold: 0.80,
        })
    }

    pub fn total_storage_bytes(&self) -> usize {
        self.clusters.iter().map(|c| c.data.len()).sum()
    }

    pub fn search(
        &self,
        raw_query: &[f32],
        k: usize,
        scratch: &mut SearchScratch,
    ) -> Vec<(usize, f32)> {
        debug_assert_eq!(raw_query.len(), self.dims as usize);
        scratch.norm_float_query.resize(self.dims as usize, 0.0);
        scratch.rotated_query.resize(self.dims as usize, 0.0);
        normalize_query(raw_query, scratch.norm_float_query.as_mut_slice());
        self.pruner.preprocess_query(
            scratch.norm_float_query.as_slice(),
            scratch.rotated_query.as_mut_slice(),
        );

        scratch.quantized_query.resize(self.dims as usize, 0);
        quantize_embedding(
            scratch.rotated_query.as_slice(),
            self.params,
            scratch.quantized_query.as_mut_slice(),
        );

        let qprep = scratch.quantized_query.as_slice();
        let max_cap = self.max_cluster_capacity().max(1);
        scratch.pruning_distances.resize(max_cap, 0);
        scratch.pruning_positions.resize(max_cap, 0);

        let mut heap = BinaryHeap::<KnCand>::new();

        for cluster in &self.clusters {
            let n_emb = cluster.num_embeddings;
            let stride = cluster.max_capacity;
            if n_emb == 0 {
                continue;
            }

            let sl_dist = &mut scratch.pruning_distances[..max_cap];
            let sl_pos = &mut scratch.pruning_positions[..max_cap];

            if heap.len() < k {
                start_cluster(
                    self,
                    qprep,
                    &cluster.data,
                    n_emb,
                    stride,
                    k as u32,
                    &cluster.corpus_indices,
                    sl_dist,
                    sl_pos,
                    &mut heap,
                );
                continue;
            }

            let tuples_needed_exit =
                ((self.selectivity_threshold * n_emb as f32).ceil() as usize).max(1);
            let mut pruning_threshold = DistanceU32::MAX;
            let mut current_dimension_idx: u32 = 0;
            let mut n_vectors_not_pruned = 0usize;

            warmup(
                self,
                qprep,
                &cluster.data,
                n_emb,
                stride,
                k as u32,
                tuples_needed_exit,
                sl_dist,
                &mut pruning_threshold,
                &mut heap,
                &mut current_dimension_idx,
                &mut n_vectors_not_pruned,
            );

            prune(
                self,
                qprep,
                &cluster.data,
                n_emb,
                stride,
                k as u32,
                sl_pos,
                sl_dist,
                &mut pruning_threshold,
                &mut heap,
                &mut current_dimension_idx,
                &mut n_vectors_not_pruned,
            );

            if n_vectors_not_pruned > 0 {
                merge_into_heap(
                    &cluster.corpus_indices,
                    n_vectors_not_pruned,
                    k,
                    sl_pos,
                    sl_dist,
                    self.inverse_quantization_scale_squared,
                    &mut heap,
                );
            }
        }

        build_result_set_from_heap(k, heap)
    }
}

pub struct SearchScratch {
    pub norm_float_query: Vec<f32>,
    pub rotated_query: Vec<f32>,
    pub quantized_query: Vec<u8>,
    pub pruning_distances: Vec<DistanceU32>,
    pub pruning_positions: Vec<u32>,
}

impl Default for SearchScratch {
    fn default() -> Self {
        Self {
            norm_float_query: Vec::new(),
            rotated_query: Vec::new(),
            quantized_query: Vec::new(),
            pruning_distances: Vec::new(),
            pruning_positions: Vec::new(),
        }
    }
}

#[inline]
fn threshold_u32(
    index: &FlatPdxU8SearchIndex,
    heap: &BinaryHeap<KnCand>,
    current_dimension_idx: u32,
) -> DistanceU32 {
    let worst = heap.peek().unwrap().dist;
    let ratio = index.pruner.pruning_distance_ratio(current_dimension_idx);
    let float_threshold = worst * ratio;
    let scaled = float_threshold * index.quantization_scale_squared;
    if scaled >= DistanceU32::MAX as f32 {
        DistanceU32::MAX
    } else {
        scaled as DistanceU32
    }
}

fn reset_pruning_distances(slice: &mut [DistanceU32], n: usize) {
    slice[..n].fill(0);
}

fn evaluate_pruning_predicate_scalar(
    n_vectors: usize,
    pruning_distances: &[DistanceU32],
    pruning_threshold: DistanceU32,
) -> u32 {
    let mut n_pruned = 0u32;
    for i in 0..n_vectors {
        if pruning_distances[i] >= pruning_threshold {
            n_pruned += 1;
        }
    }
    n_pruned
}

fn init_positions_array(
    n_vectors: usize,
    pruning_positions: &mut [u32],
    pruning_threshold: DistanceU32,
    pruning_distances: &[DistanceU32],
) -> usize {
    let mut n_vectors_not_pruned = 0usize;
    for vector_idx in 0..n_vectors {
        pruning_positions[n_vectors_not_pruned] = vector_idx as u32;
        if pruning_distances[vector_idx] < pruning_threshold {
            n_vectors_not_pruned += 1;
        }
    }
    n_vectors_not_pruned
}

fn evaluate_pruning_predicate_positions(
    n_vectors: usize,
    pruning_positions: &mut [u32],
    pruning_threshold: DistanceU32,
    pruning_distances: &[DistanceU32],
) -> usize {
    let mut n_vectors_not_pruned = 0usize;
    for vector_idx in 0..n_vectors {
        let idx = pruning_positions[vector_idx] as usize;
        pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
        if pruning_distances[idx] < pruning_threshold {
            n_vectors_not_pruned += 1;
        }
    }
    n_vectors_not_pruned
}

fn start_cluster(
    index: &FlatPdxU8SearchIndex,
    query: &[u8],
    data: &[u8],
    n_vectors: usize,
    stride: usize,
    k: u32,
    corpus_indices: &[u32],
    pruning_distances: &mut [DistanceU32],
    _pruning_positions: &mut [u32],
    heap: &mut BinaryHeap<KnCand>,
) {
    reset_pruning_distances(pruning_distances, n_vectors);
    vertical_full(
        query,
        data,
        n_vectors,
        stride,
        0,
        index.vertical_dimensions as usize,
        pruning_distances,
    );

    let vd = index.vertical_dimensions as usize;
    let hd = index.horizontal_dimensions as usize;
    let mut horizontal_dimension = 0usize;
    while horizontal_dimension < hd {
        let offset_data = vd * stride + horizontal_dimension * stride;
        let offset_query = vd + horizontal_dimension;
        for vector_idx in 0..n_vectors {
            let data_pos = offset_data + vector_idx * super::layout::H_DIM_SIZE as usize;
            pruning_distances[vector_idx] =
                pruning_distances[vector_idx].saturating_add(horizontal_chunk(
                    &query[offset_query..offset_query + super::layout::H_DIM_SIZE as usize],
                    &data[data_pos..data_pos + super::layout::H_DIM_SIZE as usize],
                    super::layout::H_DIM_SIZE as usize,
                ));
        }
        horizontal_dimension += super::layout::H_DIM_SIZE as usize;
    }

    let max_possible_k = (k as usize).saturating_sub(heap.len()).min(n_vectors);
    let mut order: Vec<usize> = (0..n_vectors).collect();
    order.sort_by_key(|&i| pruning_distances[i]);
    for idx in order.into_iter().take(max_possible_k) {
        let dist_f = pruning_distances[idx] as f32 * index.inverse_quantization_scale_squared;
        heap.push(KnCand {
            dist: dist_f,
            corpus_idx: corpus_indices[idx],
        });
    }
}

#[allow(unused_variables)]
fn warmup(
    index: &FlatPdxU8SearchIndex,
    query: &[u8],
    data: &[u8],
    n_vectors: usize,
    stride: usize,
    k: u32,
    tuples_needed_exit: usize,
    pruning_distances: &mut [DistanceU32],
    pruning_threshold: &mut DistanceU32,
    heap: &mut BinaryHeap<KnCand>,
    current_dimension_idx: &mut u32,
    n_vectors_not_pruned: &mut usize,
) {
    let vd = index.vertical_dimensions as usize;
    *current_dimension_idx = 0;
    let mut cur_subgrouping_size_idx = 0usize;
    reset_pruning_distances(pruning_distances, n_vectors);

    let mut n_tuples_to_prune = 0u32;
    *pruning_threshold = threshold_u32(index, heap, *current_dimension_idx);

    while (n_tuples_to_prune as usize) < tuples_needed_exit
        && (*current_dimension_idx as usize) < vd
    {
        let fetch_sz = DIMENSIONS_FETCHING_SIZES
            [cur_subgrouping_size_idx.min(DIMENSIONS_FETCHING_SIZES.len() - 1)];
        let last_dimension_to_fetch = ((*current_dimension_idx as usize) + fetch_sz).min(vd);
        vertical_full(
            query,
            data,
            n_vectors,
            stride,
            *current_dimension_idx as usize,
            last_dimension_to_fetch,
            pruning_distances,
        );
        *current_dimension_idx = last_dimension_to_fetch as u32;
        cur_subgrouping_size_idx += 1;
        *pruning_threshold = threshold_u32(index, heap, *current_dimension_idx);
        n_tuples_to_prune = 0;
        n_tuples_to_prune +=
            evaluate_pruning_predicate_scalar(n_vectors, pruning_distances, *pruning_threshold);
    }
    *n_vectors_not_pruned = 0;
}

fn prune(
    index: &FlatPdxU8SearchIndex,
    query: &[u8],
    data: &[u8],
    n_vectors: usize,
    stride: usize,
    k: u32,
    pruning_positions: &mut [u32],
    pruning_distances: &mut [DistanceU32],
    pruning_threshold: &mut DistanceU32,
    heap: &mut BinaryHeap<KnCand>,
    current_dimension_idx: &mut u32,
    n_vectors_not_pruned: &mut usize,
) {
    let _ = k;
    *pruning_threshold = threshold_u32(index, heap, *current_dimension_idx);
    *n_vectors_not_pruned = init_positions_array(
        n_vectors,
        pruning_positions,
        *pruning_threshold,
        pruning_distances,
    );

    let vd = index.vertical_dimensions as usize;
    let hd = index.horizontal_dimensions as usize;
    let num_dims = index.dims as usize;

    let mut current_vertical_dimension = *current_dimension_idx as usize;
    let mut current_horizontal_dimension = 0usize;

    while hd > 0 && *n_vectors_not_pruned > 0 && current_horizontal_dimension < hd {
        let cur_n = *n_vectors_not_pruned;
        let offset_data = vd * stride + current_horizontal_dimension * stride;
        let offset_query = vd + current_horizontal_dimension;

        for vector_idx in 0..cur_n {
            let v_idx = pruning_positions[vector_idx] as usize;
            let data_pos = offset_data + v_idx * super::layout::H_DIM_SIZE as usize;
            pruning_distances[v_idx] = pruning_distances[v_idx].saturating_add(horizontal_chunk(
                &query[offset_query..offset_query + super::layout::H_DIM_SIZE as usize],
                &data[data_pos..data_pos + super::layout::H_DIM_SIZE as usize],
                super::layout::H_DIM_SIZE as usize,
            ));
        }

        current_horizontal_dimension += super::layout::H_DIM_SIZE as usize;
        *current_dimension_idx += super::layout::H_DIM_SIZE;
        debug_assert_eq!(
            *current_dimension_idx as usize,
            current_vertical_dimension + current_horizontal_dimension
        );
        *pruning_threshold = threshold_u32(index, heap, *current_dimension_idx);
        *n_vectors_not_pruned = evaluate_pruning_predicate_positions(
            cur_n,
            pruning_positions,
            *pruning_threshold,
            pruning_distances,
        );
    }

    while *n_vectors_not_pruned > 0 && current_vertical_dimension < vd {
        let cur_n = *n_vectors_not_pruned;
        let last_dimension_to_test_idx =
            (current_vertical_dimension + super::layout::H_DIM_SIZE as usize).min(vd);

        vertical_pruning(
            query,
            data,
            cur_n,
            stride,
            current_vertical_dimension,
            last_dimension_to_test_idx,
            pruning_distances,
            &pruning_positions[..cur_n],
        );

        *current_dimension_idx = ((*current_dimension_idx as usize)
            + super::layout::H_DIM_SIZE as usize)
            .min(num_dims) as u32;
        current_vertical_dimension =
            (current_vertical_dimension + super::layout::H_DIM_SIZE as usize).min(vd);

        debug_assert_eq!(
            *current_dimension_idx as usize,
            current_vertical_dimension + current_horizontal_dimension
        );
        *pruning_threshold = threshold_u32(index, heap, *current_dimension_idx);
        *n_vectors_not_pruned = evaluate_pruning_predicate_positions(
            cur_n,
            pruning_positions,
            *pruning_threshold,
            pruning_distances,
        );

        if *current_dimension_idx as usize == num_dims {
            break;
        }
    }
}

fn merge_into_heap(
    corpus_indices: &[u32],
    n_vectors_not_pruned: usize,
    k: usize,
    pruning_positions: &[u32],
    pruning_distances: &[DistanceU32],
    inverse_qss: f32,
    heap: &mut BinaryHeap<KnCand>,
) {
    for position_idx in 0..n_vectors_not_pruned {
        let idx = pruning_positions[position_idx] as usize;
        let current_distance = pruning_distances[idx] as f32 * inverse_qss;
        heap_maybe_push(
            heap,
            k,
            KnCand {
                dist: current_distance,
                corpus_idx: corpus_indices[idx],
            },
        );
    }
}

fn build_result_set_from_heap(k: usize, mut heap: BinaryHeap<KnCand>) -> Vec<(usize, f32)> {
    let mut tmp = Vec::new();
    while let Some(c) = heap.pop() {
        tmp.push((c.corpus_idx as usize, c.dist));
    }
    tmp.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    tmp.truncate(k.min(tmp.len()));
    tmp
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn normalize_vec(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in v {
            *x /= n;
        }
    }

    fn quant_l2_sq(q: &[u8], d: &[u8]) -> u32 {
        q.iter()
            .zip(d.iter())
            .map(|(&a, &b)| {
                let di = i32::from(a) - i32::from(b);
                (di * di) as u32
            })
            .sum()
    }

    /// One chunk ⇒ only `start_cluster` runs (heap never prefilled across clusters): ranking must match full quantized L² in the ADSampling basis.
    #[test]
    fn single_chunk_search_matches_brute_quantized_rotated_l2() {
        let dims = 32usize;
        let n_docs = 80usize;
        let mut rng = StdRng::seed_from_u64(999);
        let mut docs = Vec::with_capacity(n_docs);
        for _ in 0..n_docs {
            let mut v = vec![0f32; dims];
            for x in &mut v {
                *x = rng.gen::<f32>() * 2.0 - 1.0;
            }
            normalize_vec(&mut v);
            docs.push(v);
        }

        let idx = FlatPdxU8SearchIndex::encode_chunked(&docs, n_docs, 42).unwrap();
        let k = 7usize;
        let query = docs[31].clone();

        let mut scratch = SearchScratch::default();
        let got = idx.search(&query, k, &mut scratch);

        scratch.norm_float_query.resize(dims, 0.);
        scratch.rotated_query.resize(dims, 0.);
        normalize_query(&query, &mut scratch.norm_float_query);
        idx.pruner.preprocess_query(
            scratch.norm_float_query.as_slice(),
            scratch.rotated_query.as_mut_slice(),
        );
        let mut qcode = vec![0u8; dims];
        quantize_embedding(scratch.rotated_query.as_slice(), idx.params, &mut qcode);

        let mut rot = vec![0f32; dims];
        let mut dcode = vec![0u8; dims];
        let mut brute: Vec<(usize, u32)> = Vec::with_capacity(n_docs);
        for (i, doc) in docs.iter().enumerate() {
            idx.pruner.rotate_embedding(doc, &mut rot);
            quantize_embedding(&rot, idx.params, &mut dcode);
            brute.push((i, quant_l2_sq(&qcode, &dcode)));
        }
        brute.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let brute_ids: Vec<usize> = brute.into_iter().take(k).map(|(i, _)| i).collect();
        let search_ids: Vec<usize> = got.into_iter().map(|(i, _)| i).collect();
        assert_eq!(
            brute_ids, search_ids,
            "single-cluster Start path must equal exhaustive quantized ranking"
        );
    }
}
