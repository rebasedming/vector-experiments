//! PDX `Quantization::U8` cluster layout and global affine quantization matching
//! cwida/PDX (`include/pdx/common.hpp`, `include/pdx/layout.hpp`,
//! `include/pdx/cluster.hpp`, `include/pdx/quantizers/scalar.hpp`).
//!
//! Dimension split mirrors PDX `GetPDXDimensionSplit` in cwida/PDX
//! `include/pdx/common.hpp` (`PROPORTION_HORIZONTAL_DIM`, special case `D <= 128`,
//! 64-wide horizontal alignment, `D <= H_DIM_SIZE` all-vertical).

pub const H_DIM_SIZE: u32 = 64;
pub const U8_INTERLEAVE_SIZE: u32 = 4;
pub const MAX_U8: u8 = 255;
pub const PROPORTION_HORIZONTAL_DIM: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimensionSplit {
    pub vertical_dimensions: u32,
    pub horizontal_dimensions: u32,
}

/// Mirrors PDX `GetPDXDimensionSplit(num_dimensions)` (`common.hpp`).
#[inline]
pub fn get_pdx_dimension_split(num_dimensions: u32) -> DimensionSplit {
    let mut local_prop = PROPORTION_HORIZONTAL_DIM;
    if num_dimensions <= 128 {
        local_prop = 0.25;
    }
    let mut horizontal_d = (num_dimensions as f32 * local_prop) as u32;
    let mut vertical_d = num_dimensions - horizontal_d;
    if horizontal_d % H_DIM_SIZE > 0 {
        horizontal_d = ((horizontal_d + H_DIM_SIZE / 2) / H_DIM_SIZE) * H_DIM_SIZE;
        vertical_d = num_dimensions - horizontal_d;
    }
    if vertical_d == 0 {
        horizontal_d = H_DIM_SIZE;
        vertical_d = num_dimensions - horizontal_d;
    }
    if num_dimensions <= H_DIM_SIZE {
        horizontal_d = 0;
        vertical_d = num_dimensions;
    }
    debug_assert_eq!(horizontal_d + vertical_d, num_dimensions);
    DimensionSplit {
        vertical_dimensions: vertical_d,
        horizontal_dimensions: horizontal_d,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarQuantParams {
    pub quantization_base: f32,
    pub quantization_scale: f32,
}

/// `ScalarQuantizer<U8>::ComputeQuantizationParams` over a contiguous row-major buffer.
pub fn compute_quantization_params(embeddings: &[f32]) -> ScalarQuantParams {
    let mut global_min = f32::MAX;
    let mut global_max = f32::NEG_INFINITY;
    for &x in embeddings {
        global_min = global_min.min(x);
        global_max = global_max.max(x);
    }
    let range = global_max - global_min;
    let quantization_scale = if range > 0.0 {
        (MAX_U8 as f32) / range
    } else {
        1.0
    };
    ScalarQuantParams {
        quantization_base: global_min,
        quantization_scale,
    }
}

#[inline]
pub fn quantize_embedding(embedding: &[f32], params: ScalarQuantParams, out: &mut [u8]) {
    assert_eq!(embedding.len(), out.len());
    for (i, &x) in embedding.iter().enumerate() {
        let rounded = ((x - params.quantization_base) * params.quantization_scale).round() as i32;
        out[i] = if rounded > i32::from(MAX_U8) {
            MAX_U8
        } else if rounded < 0 {
            0
        } else {
            rounded as u8
        };
    }
}

/// PDX `Quantizer::NormalizeQuery`: L2-normalize `src` into `out`.
#[inline]
pub fn normalize_query(src: &[f32], out: &mut [f32]) {
    assert_eq!(src.len(), out.len());
    let mut sum = 0.0f32;
    for &x in src {
        sum += x * x;
    }
    if sum == 0.0 {
        out.fill(0.0);
        return;
    }
    let norm = sum.sqrt();
    for i in 0..src.len() {
        out[i] = src[i] / norm;
    }
}

#[inline]
pub fn dequantize_embedding(codes: &[u8], params: ScalarQuantParams, out: &mut [f32]) {
    assert_eq!(codes.len(), out.len());
    let inv_scale = 1.0 / params.quantization_scale;
    for i in 0..codes.len() {
        out[i] = codes[i] as f32 * inv_scale + params.quantization_base;
    }
}

/// Total packed bytes for one PDX cluster buffer: `stride * num_dimensions`.
#[inline]
pub fn cluster_buffer_bytes(stride: usize, num_dimensions: u32) -> usize {
    stride.saturating_mul(num_dimensions as usize)
}

/// Scatter one row-major `U8` embedding into the PDX U8 cluster buffer at `idx_in_cluster`.
/// `data` layout matches `Cluster<U8>::InsertEmbedding` in PDX `cluster.hpp`.
pub fn scatter_u8_embedding_into_cluster(
    data: &mut [u8],
    stride: usize,
    idx_in_cluster: usize,
    embedding: &[u8],
    num_dimensions: u32,
) {
    let split = get_pdx_dimension_split(num_dimensions);
    let vertical_d = split.vertical_dimensions;
    let horizontal_d = split.horizontal_dimensions;
    assert_eq!(embedding.len(), num_dimensions as usize);

    let mut d = 0u32;
    while d + U8_INTERLEAVE_SIZE <= vertical_d {
        let base = (d as usize) * stride + idx_in_cluster * U8_INTERLEAVE_SIZE as usize;
        data[base..base + U8_INTERLEAVE_SIZE as usize]
            .copy_from_slice(&embedding[d as usize..(d + U8_INTERLEAVE_SIZE) as usize]);
        d += U8_INTERLEAVE_SIZE;
    }
    if d < vertical_d {
        let remaining = vertical_d - d;
        let base = (d as usize) * stride + idx_in_cluster * remaining as usize;
        data[base..base + remaining as usize]
            .copy_from_slice(&embedding[d as usize..vertical_d as usize]);
    }

    let mut h_base = stride * vertical_d as usize;
    let mut j = 0u32;
    while j < horizontal_d {
        let chunk = h_base + idx_in_cluster * H_DIM_SIZE as usize;
        let src_start = (vertical_d + j) as usize;
        data[chunk..chunk + H_DIM_SIZE as usize]
            .copy_from_slice(&embedding[src_start..src_start + H_DIM_SIZE as usize]);
        h_base += stride * H_DIM_SIZE as usize;
        j += H_DIM_SIZE;
    }
}

/// Gather from PDX U8 cluster buffer → row-major `U8` (`ReadEmbeddingFromPDXBuffer`).
pub fn gather_u8_embedding_from_cluster(
    data: &[u8],
    stride: usize,
    idx_in_cluster: usize,
    out: &mut [u8],
    num_dimensions: u32,
) {
    let split = get_pdx_dimension_split(num_dimensions);
    let vertical_d = split.vertical_dimensions;
    let horizontal_d = split.horizontal_dimensions;
    assert_eq!(out.len(), num_dimensions as usize);

    let mut d = 0u32;
    while d + U8_INTERLEAVE_SIZE <= vertical_d {
        let base = (d as usize) * stride + idx_in_cluster * U8_INTERLEAVE_SIZE as usize;
        out[d as usize..(d + U8_INTERLEAVE_SIZE) as usize]
            .copy_from_slice(&data[base..base + U8_INTERLEAVE_SIZE as usize]);
        d += U8_INTERLEAVE_SIZE;
    }
    if d < vertical_d {
        let remaining = vertical_d - d;
        let base = (d as usize) * stride + idx_in_cluster * remaining as usize;
        out[d as usize..vertical_d as usize]
            .copy_from_slice(&data[base..base + remaining as usize]);
    }

    let mut h_base = stride * vertical_d as usize;
    let mut j = 0u32;
    while j < horizontal_d {
        let chunk = h_base + idx_in_cluster * H_DIM_SIZE as usize;
        let dst_start = (vertical_d + j) as usize;
        out[dst_start..dst_start + H_DIM_SIZE as usize]
            .copy_from_slice(&data[chunk..chunk + H_DIM_SIZE as usize]);
        h_base += stride * H_DIM_SIZE as usize;
        j += H_DIM_SIZE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_split_matches_pdx_static_asserts() {
        assert_eq!(
            get_pdx_dimension_split(768),
            DimensionSplit {
                vertical_dimensions: 192,
                horizontal_dimensions: 576
            }
        );
        assert_eq!(
            get_pdx_dimension_split(1024),
            DimensionSplit {
                vertical_dimensions: 256,
                horizontal_dimensions: 768
            }
        );
        assert_eq!(
            get_pdx_dimension_split(1536),
            DimensionSplit {
                vertical_dimensions: 384,
                horizontal_dimensions: 1152
            }
        );
        assert_eq!(
            get_pdx_dimension_split(128),
            DimensionSplit {
                vertical_dimensions: 64,
                horizontal_dimensions: 64
            }
        );
        assert_eq!(
            get_pdx_dimension_split(100),
            DimensionSplit {
                vertical_dimensions: 100,
                horizontal_dimensions: 0
            }
        );
        assert_eq!(
            get_pdx_dimension_split(1028),
            DimensionSplit {
                vertical_dimensions: 260,
                horizontal_dimensions: 768
            }
        );
    }

    #[test]
    fn scatter_gather_roundtrip_random_u8() {
        let dims = 768u32;
        let n = 17usize;
        let stride = n;
        let mut buf = vec![0u8; cluster_buffer_bytes(stride, dims)];
        let mut emb = vec![0u8; dims as usize];
        let mut tmp = vec![0u8; dims as usize];
        for i in 0..n {
            for (j, b) in emb.iter_mut().enumerate() {
                *b = ((i + j) % 251) as u8;
            }
            scatter_u8_embedding_into_cluster(&mut buf, stride, i, &emb, dims);
        }
        for i in 0..n {
            gather_u8_embedding_from_cluster(&buf, stride, i, &mut tmp, dims);
            for (j, b) in emb.iter_mut().enumerate() {
                *b = ((i + j) % 251) as u8;
            }
            assert_eq!(tmp, emb, "slot {i}");
        }
    }

    #[test]
    fn quantize_dequantize_identity_midrange() {
        let v = vec![0.0f32, 0.5, 1.0];
        let mut codes = vec![0u8; 3];
        let mut back = vec![0f32; 3];
        let p = compute_quantization_params(&v);
        quantize_embedding(&v, p, &mut codes);
        dequantize_embedding(&codes, p, &mut back);
        for i in 0..3 {
            let err = (back[i] - v[i]).abs();
            assert!(
                err < 2.0 / 255.0,
                "i={i} got {} want {} err {}",
                back[i],
                v[i],
                err
            );
        }
    }
}
