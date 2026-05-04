//! Scalar `DistanceComputer<L2SQ, U8>` from cwida/PDX `scalar_computers.hpp`.

use super::layout::{H_DIM_SIZE, U8_INTERLEAVE_SIZE};

pub type DistanceU32 = u32;

#[inline]
fn sq_diff_u8(a: u8, b: u8) -> u32 {
    let d = i32::from(a) - i32::from(b);
    (d * d) as u32
}

/// PDX `ScalarComputer<L2SQ,U8>::Horizontal`.
pub fn horizontal_chunk(query: &[u8], data: &[u8], num_dimensions: usize) -> DistanceU32 {
    debug_assert_eq!(query.len(), num_dimensions);
    debug_assert_eq!(data.len(), num_dimensions);
    let mut distance = 0u32;
    for i in 0..num_dimensions {
        distance += sq_diff_u8(query[i], data[i]);
    }
    distance
}

/// PDX `ScalarComputer<L2SQ,U8>::Vertical` (`SKIP_PRUNED = false`).
#[allow(clippy::too_many_arguments)]
pub fn vertical_full(
    query: &[u8],
    data: &[u8],
    n_vectors: usize,
    stride: usize,
    start_dimension: usize,
    end_dimension: usize,
    distances_p: &mut [DistanceU32],
) {
    vertical_inner::<false>(
        query,
        data,
        n_vectors,
        stride,
        start_dimension,
        end_dimension,
        distances_p,
        None,
    );
}

/// PDX `ScalarComputer<L2SQ,U8>::Vertical` (`SKIP_PRUNED = true`).
#[allow(clippy::too_many_arguments)]
pub fn vertical_pruning(
    query: &[u8],
    data: &[u8],
    n_vectors_not_pruned: usize,
    stride: usize,
    start_dimension: usize,
    end_dimension: usize,
    distances_p: &mut [DistanceU32],
    pruning_positions: &[u32],
) {
    vertical_inner::<true>(
        query,
        data,
        n_vectors_not_pruned,
        stride,
        start_dimension,
        end_dimension,
        distances_p,
        Some(pruning_positions),
    );
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn vertical_inner<const SKIP_PRUNED: bool>(
    query: &[u8],
    data: &[u8],
    n_vectors: usize,
    stride: usize,
    start_dimension: usize,
    end_dimension: usize,
    distances_p: &mut [DistanceU32],
    pruning_positions: Option<&[u32]>,
) {
    let mut dim_idx = start_dimension;
    while dim_idx + U8_INTERLEAVE_SIZE as usize <= end_dimension {
        let dimension_idx = dim_idx;
        let offset_to_dimension_start = dimension_idx * stride;
        for i in 0..n_vectors {
            let vector_idx = if SKIP_PRUNED {
                pruning_positions.unwrap()[i] as usize
            } else {
                i
            };
            let row_off = offset_to_dimension_start + vector_idx * U8_INTERLEAVE_SIZE as usize;
            let da = i32::from(query[dimension_idx]) - i32::from(data[row_off]);
            let db = i32::from(query[dimension_idx + 1]) - i32::from(data[row_off + 1]);
            let dc = i32::from(query[dimension_idx + 2]) - i32::from(data[row_off + 2]);
            let dd = i32::from(query[dimension_idx + 3]) - i32::from(data[row_off + 3]);
            distances_p[vector_idx] = distances_p[vector_idx]
                .saturating_add(((da * da) + (db * db) + (dc * dc) + (dd * dd)) as u32);
        }
        dim_idx += U8_INTERLEAVE_SIZE as usize;
    }
    if dim_idx < end_dimension {
        let remaining = end_dimension - dim_idx;
        let offset = dim_idx * stride;
        for i in 0..n_vectors {
            let vector_idx = if SKIP_PRUNED {
                pruning_positions.unwrap()[i] as usize
            } else {
                i
            };
            for k in 0..remaining {
                let diff = i32::from(query[dim_idx + k])
                    - i32::from(data[offset + vector_idx * remaining + k]);
                distances_p[vector_idx] =
                    distances_p[vector_idx].saturating_add((diff * diff) as u32);
            }
        }
    }
}

/// Horizontal strip inside PDX horizontal block (`H_DIM_SIZE == 64`).
#[inline]
pub fn horizontal_strip(
    query_tail: &[u8],
    data_strip: &[u8],
    horizontal_extent: usize,
) -> DistanceU32 {
    horizontal_chunk(
        query_tail,
        data_strip,
        horizontal_extent.min(H_DIM_SIZE as usize),
    )
}
