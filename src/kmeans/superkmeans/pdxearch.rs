//! PDXearch top-1 search with ADSampling pruning.
//!
//! Mirrors `pdx/pdxearch.h`. The C++ keeps a thread-local
//! `pruning_positions[VECTOR_CHUNK_SIZE]` buffer; we expect the caller to
//! supply the equivalent scratch per thread, since rayon worker threads will
//! call this concurrently.

use super::common::{KnnCandidate, H_DIM_SIZE};
use super::computers::{init_positions_array, squared_l2_horizontal};
use super::layout::{IndexPdxIvf, PdxCluster};
use super::rotation::AdSamplingPruner;

/// View into one cluster's PDX data.
struct ClusterData<'a> {
    data: &'a [f32],
    indices: &'a [u32],
    aux: &'a [f32],
    num_embeddings: usize,
}

fn slice_cluster<'a>(
    cluster: &PdxCluster,
    pdx_data: &'a [f32],
    pdx_indices: &'a [u32],
    aux: &'a [f32],
    d: usize,
    vertical_d: usize,
) -> ClusterData<'a> {
    let n = cluster.num_embeddings as usize;
    ClusterData {
        data: &pdx_data[cluster.data_offset..cluster.data_offset + n * d],
        indices: &pdx_indices[cluster.indices_offset..cluster.indices_offset + n],
        aux: &aux[cluster.aux_offset..cluster.aux_offset + n * vertical_d],
        num_embeddings: n,
    }
}

#[allow(clippy::too_many_arguments)]
fn prune(
    pruner: &AdSamplingPruner,
    cluster: &ClusterData,
    query: &[f32],
    pruning_positions: &mut [u32],
    pruning_distances: &mut [f32],
    best: &mut KnnCandidate,
    n_not_pruned: &mut usize,
    current_dimension_idx: &mut u32,
    prev_top_1: u32,
    num_horizontal_dimensions: u32,
    num_vertical_dimensions: u32,
    num_dimensions: u32,
) -> usize {
    let n_vectors = cluster.num_embeddings;

    let mut threshold = pruner.pruning_threshold(best, *current_dimension_idx);
    *n_not_pruned = init_positions_array(
        pruning_positions,
        pruning_distances,
        threshold,
        n_vectors,
    );
    let initial_not_pruned = *n_not_pruned;
    if *n_not_pruned == 1
        && cluster.indices[pruning_positions[0] as usize] == prev_top_1
    {
        *n_not_pruned = 0;
        return initial_not_pruned;
    }

    let mut cur_n_not_pruned;
    let current_vertical_dimension = *current_dimension_idx as usize;
    let mut current_horizontal_dimension = 0usize;

    // Horizontal blocks (H_DIM_SIZE columns at a time).
    while num_horizontal_dimensions > 0
        && *n_not_pruned > 0
        && current_horizontal_dimension < num_horizontal_dimensions as usize
    {
        cur_n_not_pruned = *n_not_pruned;
        let vert_block = num_vertical_dimensions as usize * n_vectors;
        let horiz_block_off = current_horizontal_dimension * n_vectors;
        let offset_data = vert_block + horiz_block_off;
        let offset_query = num_vertical_dimensions as usize + current_horizontal_dimension;
        for k in 0..*n_not_pruned {
            let v_idx = pruning_positions[k] as usize;
            let data_pos = offset_data + v_idx * H_DIM_SIZE;
            let q = &query[offset_query..offset_query + H_DIM_SIZE];
            let d_slice = &cluster.data[data_pos..data_pos + H_DIM_SIZE];
            pruning_distances[v_idx] += squared_l2_horizontal(q, d_slice);
        }
        current_horizontal_dimension += H_DIM_SIZE;
        *current_dimension_idx += H_DIM_SIZE as u32;
        threshold = pruner.pruning_threshold(best, *current_dimension_idx);
        debug_assert_eq!(
            *current_dimension_idx as usize,
            current_vertical_dimension + current_horizontal_dimension
        );
        // Re-evaluate predicate, compacting in place over the *previous* prefix.
        evaluate_predicate(
            pruning_positions,
            pruning_distances,
            threshold,
            cur_n_not_pruned,
            n_not_pruned,
        );
    }

    // Trailing vertical dims (after partial_d) via aux row-major buffer.
    if *n_not_pruned > 0 && current_vertical_dimension < num_vertical_dimensions as usize {
        cur_n_not_pruned = *n_not_pruned;
        let dimensions_left = num_vertical_dimensions as usize - current_vertical_dimension;
        let offset_query = current_vertical_dimension;
        for k in 0..*n_not_pruned {
            let v_idx = pruning_positions[k] as usize;
            let data_pos = v_idx * num_vertical_dimensions as usize + current_vertical_dimension;
            let q = &query[offset_query..offset_query + dimensions_left];
            let d_slice = &cluster.aux[data_pos..data_pos + dimensions_left];
            pruning_distances[v_idx] += squared_l2_horizontal(q, d_slice);
        }
        *current_dimension_idx = num_dimensions;
        threshold = pruner.pruning_threshold(best, *current_dimension_idx);
        evaluate_predicate(
            pruning_positions,
            pruning_distances,
            threshold,
            cur_n_not_pruned,
            n_not_pruned,
        );
    }

    initial_not_pruned
}

fn evaluate_predicate(
    pruning_positions: &mut [u32],
    pruning_distances: &[f32],
    threshold: f32,
    cur_count: usize,
    n_not_pruned: &mut usize,
) {
    let mut new_count = 0usize;
    for k in 0..cur_count {
        let v = pruning_positions[k];
        pruning_positions[new_count] = v;
        if pruning_distances[v as usize] < threshold {
            new_count += 1;
        }
    }
    *n_not_pruned = new_count;
}

fn set_best_candidate(
    indices: &[u32],
    pruning_positions: &[u32],
    pruning_distances: &[f32],
    n_not_pruned: usize,
    best: &mut KnnCandidate,
) {
    for k in 0..n_not_pruned {
        let pos = pruning_positions[k] as usize;
        let dist = pruning_distances[pos];
        if dist < best.distance {
            best.distance = dist;
            best.index = indices[pos];
        }
    }
}

/// Mirrors `PDXearch::Top1PartialSearchWithThresholdAndPartialDistances`.
///
/// `partial_pruning_distances` is the row of the GEMM result for one query
/// (length `≥ sum of cluster sizes in [start_cluster, end_cluster)`). On
/// entry, distances cover the first `computed_distance_until` dimensions
/// (squared L2). They are mutated in place as more dims are accumulated.
#[allow(clippy::too_many_arguments)]
pub fn top1_partial_search(
    pruner: &AdSamplingPruner,
    index: &IndexPdxIvf,
    pdx_data: &[f32],
    pdx_indices: &[u32],
    aux: &[f32],
    query: &[f32],
    prev_pruning_threshold: f32,
    prev_top_1: u32,
    partial_pruning_distances: &mut [f32],
    computed_distance_until: u32,
    start_cluster: usize,
    end_cluster: usize,
    pruning_positions_scratch: &mut [u32],
    initial_not_pruned_accum: &mut usize,
) -> KnnCandidate {
    let d = index.num_dimensions as usize;
    let vertical_d = index.num_vertical_dimensions as usize;
    let mut top = KnnCandidate {
        index: prev_top_1,
        distance: prev_pruning_threshold,
    };

    let mut data_offset = 0usize;
    for cluster_idx in start_cluster..end_cluster {
        let cluster = &index.clusters[cluster_idx];
        let cdata = slice_cluster(cluster, pdx_data, pdx_indices, aux, d, vertical_d);
        let n = cdata.num_embeddings;
        let pruning_distances = &mut partial_pruning_distances[data_offset..data_offset + n];
        data_offset += n;

        let mut n_not_pruned = 0usize;
        let mut current_dim = computed_distance_until;
        let initial = prune(
            pruner,
            &cdata,
            query,
            pruning_positions_scratch,
            pruning_distances,
            &mut top,
            &mut n_not_pruned,
            &mut current_dim,
            prev_top_1,
            index.num_horizontal_dimensions,
            index.num_vertical_dimensions,
            index.num_dimensions,
        );
        *initial_not_pruned_accum += initial;

        if n_not_pruned > 0 {
            set_best_candidate(
                cdata.indices,
                pruning_positions_scratch,
                pruning_distances,
                n_not_pruned,
                &mut top,
            );
        }
    }

    top
}
