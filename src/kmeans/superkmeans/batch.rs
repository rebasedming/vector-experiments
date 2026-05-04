//! Batch distance kernels: GEMM-only and GEMM + PDX pruning.
//!
//! Mirrors `distance_computers/batch_computers.h`. The Apple AMX
//! `MINI_BATCH_SIZE` partitioning of the GEMM is intentionally collapsed into
//! a single sgemm — the result is bitwise-identical, only the BLAS scheduling
//! differs, and `matrixmultiply` has its own internal blocking.

use rayon::prelude::*;

use super::blas::{sgemm_row_major, Trans};
use super::common::{KnnCandidate, MINI_BATCH_SIZE, VECTOR_CHUNK_SIZE, X_BATCH_SIZE, Y_BATCH_SIZE};
use super::computers::squared_l2_horizontal;
use super::layout::IndexPdxIvf;
use super::pdxearch::top1_partial_search;
use super::rotation::AdSamplingPruner;

/// Compute `tmp[r,c] = dot(x[i+r, 0..k], y[j+c, 0..k])` where
/// `k = if partial_d > 0 && partial_d < d { partial_d } else { d }`.
///
/// On macOS, partitions the X-batch into `MINI_BATCH_SIZE` chunks and
/// dispatches one Accelerate sgemm per chunk in parallel — matches the C++
/// `#if defined(__APPLE__)` AMX scheduling pattern in `BlasMatrixMultiplication`.
fn blas_matmul(
    batch_x: &[f32],
    batch_y: &[f32],
    batch_n_x: usize,
    batch_n_y: usize,
    d: usize,
    partial_d: usize,
    tmp_distances: &mut [f32],
) {
    let k = if partial_d > 0 && partial_d < d {
        partial_d
    } else {
        d
    };

    if cfg!(target_os = "macos") {
        let active = &mut tmp_distances[..batch_n_x * batch_n_y];
        active
            .par_chunks_mut(MINI_BATCH_SIZE * batch_n_y)
            .enumerate()
            .for_each(|(chunk_idx, tmp_chunk)| {
                let r0 = chunk_idx * MINI_BATCH_SIZE;
                let mini_n_x = (batch_n_x - r0).min(MINI_BATCH_SIZE);
                let x_chunk = &batch_x[r0 * d..(r0 + mini_n_x) * d];
                sgemm_row_major(
                    Trans::No,
                    Trans::Yes,
                    mini_n_x,
                    batch_n_y,
                    k,
                    1.0,
                    x_chunk,
                    d,
                    batch_y,
                    d,
                    0.0,
                    tmp_chunk,
                    batch_n_y,
                );
            });
    } else {
        sgemm_row_major(
            Trans::No,
            Trans::Yes,
            batch_n_x,
            batch_n_y,
            k,
            1.0,
            batch_x,
            d,
            batch_y,
            d,
            0.0,
            tmp_distances,
            batch_n_y,
        );
    }
}

/// Mirrors `BatchComputer<l2, f32>::FindNearestNeighbor`. Top-1 nearest
/// neighbour, full-distance, no pruning.
#[allow(clippy::too_many_arguments)]
pub fn find_nearest_neighbor(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    norms_x: &[f32],
    norms_y: &[f32],
    out_knn: &mut [u32],
    out_distances: &mut [f32],
    tmp_distances_buf: &mut [f32],
) {
    out_distances.fill(f32::MAX);

    let mut i = 0usize;
    while i < n_x {
        let batch_n_x = X_BATCH_SIZE.min(n_x - i);
        let batch_x = &x[i * d..(i + batch_n_x) * d];

        let mut j = 0usize;
        while j < n_y {
            let batch_n_y = Y_BATCH_SIZE.min(n_y - j);
            let batch_y = &y[j * d..(j + batch_n_y) * d];
            let tmp = &mut tmp_distances_buf[..batch_n_x * batch_n_y];

            blas_matmul(batch_x, batch_y, batch_n_x, batch_n_y, d, 0, tmp);

            // Convert dot products to squared L2 and reduce per row.
            let updates: Vec<(usize, f32, u32)> = (0..batch_n_x)
                .into_par_iter()
                .map(|r| {
                    let i_idx = i + r;
                    let norm_x_i = norms_x[i_idx];
                    let row_start = r * batch_n_y;
                    let row = &tmp[row_start..row_start + batch_n_y];
                    let mut best_c = 0usize;
                    let mut best_val = f32::MAX;
                    for c in 0..batch_n_y {
                        let dist = -2.0 * row[c] + norm_x_i + norms_y[j + c];
                        if dist < best_val {
                            best_val = dist;
                            best_c = c;
                        }
                    }
                    (i_idx, best_val, (j + best_c) as u32)
                })
                .collect();

            for (i_idx, val, knn_idx) in updates {
                if val < out_distances[i_idx] {
                    out_distances[i_idx] = val.max(0.0);
                    out_knn[i_idx] = knn_idx;
                }
            }
            j += Y_BATCH_SIZE;
        }
        i += X_BATCH_SIZE;
    }
}

/// Mirrors `BatchComputer<l2, f32>::FindKNearestNeighbors`. Used for GT
/// computation and recall measurement.
#[allow(clippy::too_many_arguments)]
pub fn find_k_nearest_neighbors(
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    norms_x: &[f32],
    norms_y: &[f32],
    k: usize,
    out_knn: &mut [u32],
    out_distances: &mut [f32],
    tmp_distances_buf: &mut [f32],
) {
    out_distances.fill(f32::MAX);
    out_knn.fill(u32::MAX);

    let mut i = 0usize;
    while i < n_x {
        let batch_n_x = X_BATCH_SIZE.min(n_x - i);
        let batch_x = &x[i * d..(i + batch_n_x) * d];

        let mut j = 0usize;
        while j < n_y {
            let batch_n_y = Y_BATCH_SIZE.min(n_y - j);
            let batch_y = &y[j * d..(j + batch_n_y) * d];
            let tmp = &mut tmp_distances_buf[..batch_n_x * batch_n_y];

            blas_matmul(batch_x, batch_y, batch_n_x, batch_n_y, d, 0, tmp);

            // Per-row partial sort merging existing top-k with new batch.
            // We collect updates first to keep the borrow non-overlapping.
            let updates: Vec<(usize, Vec<(f32, u32)>)> = (0..batch_n_x)
                .into_par_iter()
                .map(|r| {
                    let i_idx = i + r;
                    let norm_x_i = norms_x[i_idx];
                    let row_start = r * batch_n_y;
                    let row = &tmp[row_start..row_start + batch_n_y];

                    let prev = &out_distances[i_idx * k..(i_idx + 1) * k];
                    let prev_idx = &out_knn[i_idx * k..(i_idx + 1) * k];

                    let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(k + batch_n_y);
                    for ki in 0..k {
                        if prev[ki] < f32::MAX {
                            candidates.push((prev[ki], prev_idx[ki]));
                        }
                    }
                    for c in 0..batch_n_y {
                        let dist = -2.0 * row[c] + norm_x_i + norms_y[j + c];
                        candidates.push((dist, (j + c) as u32));
                    }
                    candidates.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.truncate(k);
                    (i_idx, candidates)
                })
                .collect();

            for (i_idx, candidates) in updates {
                let actual = candidates.len();
                for ki in 0..actual {
                    out_distances[i_idx * k + ki] = candidates[ki].0.max(0.0);
                    out_knn[i_idx * k + ki] = candidates[ki].1;
                }
                for ki in actual..k {
                    out_distances[i_idx * k + ki] = f32::MAX;
                    out_knn[i_idx * k + ki] = u32::MAX;
                }
            }

            j += Y_BATCH_SIZE;
        }
        i += X_BATCH_SIZE;
    }
}

/// Bundle of inputs for `find_nearest_neighbor_with_pruning`: the PDX
/// representation of the centroids and the auxiliary horizontal copy of the
/// vertical dims.
pub struct PdxCentroidsRef<'a> {
    pub index: &'a IndexPdxIvf,
    pub pdx_data: &'a [f32],
    pub pdx_indices: &'a [u32],
    pub aux: &'a [f32],
}

/// Mirrors `BatchComputer<l2, f32>::FindNearestNeighborWithPruning`.
///
/// `out_knn` and `out_distances` are simultaneously inputs and outputs:
/// they carry forward the previous Lloyd iteration's assignments.
#[allow(clippy::too_many_arguments)]
pub fn find_nearest_neighbor_with_pruning(
    pruner: &AdSamplingPruner,
    x: &[f32],
    y: &[f32],
    n_x: usize,
    n_y: usize,
    d: usize,
    partial_norms_x: &[f32],
    partial_norms_y: &[f32],
    out_knn: &mut [u32],
    out_distances: &mut [f32],
    tmp_distances_buf: &mut [f32],
    centroids: &PdxCentroidsRef,
    partial_d: u32,
    out_not_pruned_counts: &mut [usize],
) {
    let mut i = 0usize;
    while i < n_x {
        let batch_n_x = X_BATCH_SIZE.min(n_x - i);
        let batch_x = &x[i * d..(i + batch_n_x) * d];

        let mut j = 0usize;
        while j < n_y {
            let batch_n_y = Y_BATCH_SIZE.min(n_y - j);
            let batch_y = &y[j * d..(j + batch_n_y) * d];
            let tmp = &mut tmp_distances_buf[..batch_n_x * batch_n_y];

            blas_matmul(
                batch_x,
                batch_y,
                batch_n_x,
                batch_n_y,
                d,
                partial_d as usize,
                tmp,
            );

            let start_cluster = j / VECTOR_CHUNK_SIZE;
            let end_cluster = (j + Y_BATCH_SIZE) / VECTOR_CHUNK_SIZE;

            // Snapshots of pre-existing per-query state, since the parallel
            // closures need read access while the slices are otherwise borrowed.
            let prev_assignments: Vec<u32> = out_knn[i..i + batch_n_x].to_vec();
            let prev_distances: Vec<f32> = out_distances[i..i + batch_n_x].to_vec();

            let row_updates: Vec<(u32, f32, usize)> = tmp
                .par_chunks_mut(batch_n_y)
                .enumerate()
                .map_init(
                    || vec![0u32; VECTOR_CHUNK_SIZE],
                    |scratch, (r, row)| {
                        let i_idx = i + r;
                        let norm_x_i = partial_norms_x[i_idx];
                        for c in 0..batch_n_y {
                            row[c] = -2.0 * row[c] + norm_x_i + partial_norms_y[j + c];
                        }

                        let prev_assignment = prev_assignments[r];
                        let dist_to_prev_centroid = if j == 0 {
                            let q = &x[i_idx * d..(i_idx + 1) * d];
                            let c_off = prev_assignment as usize * d;
                            let c_vec = &y[c_off..c_off + d];
                            squared_l2_horizontal(q, c_vec)
                        } else {
                            prev_distances[r]
                        };

                        let mut local_not_pruned = 0usize;
                        let result = top1_partial_search(
                            pruner,
                            centroids.index,
                            centroids.pdx_data,
                            centroids.pdx_indices,
                            centroids.aux,
                            &x[i_idx * d..(i_idx + 1) * d],
                            dist_to_prev_centroid,
                            prev_assignment,
                            row,
                            partial_d,
                            start_cluster,
                            end_cluster,
                            scratch.as_mut_slice(),
                            &mut local_not_pruned,
                        );
                        (result.index, result.distance, local_not_pruned)
                    },
                )
                .collect();

            for (r, (knn, dist, not_pruned)) in row_updates.into_iter().enumerate() {
                let i_idx = i + r;
                out_knn[i_idx] = knn;
                out_distances[i_idx] = dist;
                out_not_pruned_counts[i_idx] += not_pruned;
            }

            j += Y_BATCH_SIZE;
        }
        i += X_BATCH_SIZE;
    }
}
