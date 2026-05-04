//! PDX in-memory layout.
//!
//! Mirrors `pdx/layout.h` and `pdx/pdx_ivf.h`. Faithful 1:1 port for
//! `(Quantization::f32, DistanceFunction::l2)`. The PDX layout splits each
//! `VECTOR_CHUNK_SIZE`-row block into:
//!   1. A "vertical" part: `vertical_d × CHUNK` (dimension-major) — currently
//!      unused at search time but still allocated/written to match the C++.
//!   2. A "horizontal" part: `horizontal_d/H_DIM_SIZE` blocks of
//!      `CHUNK × H_DIM_SIZE` (row-major), concatenated.
//!
//! Search-time access for the vertical dims goes through a separate row-major
//! "auxiliary" buffer (`aux_vertical_dimensions_in_horizontal_layout`).

use super::common::{H_DIM_SIZE, PROPORTION_HORIZONTAL_DIM, VECTOR_CHUNK_SIZE};

#[derive(Debug, Clone, Copy)]
pub struct DimensionSplit {
    pub horizontal_d: usize,
    pub vertical_d: usize,
}

/// Mirrors `PDXLayout::GetDimensionSplit`. 25% vertical / 75% horizontal,
/// rounded so `horizontal_d` is a multiple of `H_DIM_SIZE`.
pub fn get_dimension_split(d: usize) -> DimensionSplit {
    let mut local_proportion_horizontal = PROPORTION_HORIZONTAL_DIM;
    if d <= 256 {
        local_proportion_horizontal = 0.25;
    }
    let mut horizontal_d = (d as f64 * local_proportion_horizontal) as usize;
    let mut vertical_d = d - horizontal_d;
    if horizontal_d % H_DIM_SIZE > 0 {
        horizontal_d = ((horizontal_d as f64 / H_DIM_SIZE as f64).round() as usize) * H_DIM_SIZE;
        vertical_d = d - horizontal_d;
    }
    if vertical_d == 0 {
        horizontal_d = H_DIM_SIZE;
        vertical_d = d - horizontal_d;
    }
    if d <= H_DIM_SIZE {
        horizontal_d = 0;
        vertical_d = d;
    }
    DimensionSplit {
        horizontal_d,
        vertical_d,
    }
}

/// Block-local description used by `PdxearchPlan` and `Prune`.
#[derive(Clone)]
pub struct PdxCluster {
    pub num_embeddings: u32,
    /// Offset into the PDX data buffer where this block starts.
    pub data_offset: usize,
    /// Offset into the indices buffer for this block (length = num_embeddings).
    pub indices_offset: usize,
    /// Offset into the auxiliary row-major vertical buffer for this block.
    pub aux_offset: usize,
}

/// Mirrors `IndexPDXIVF<f32>`.
pub struct IndexPdxIvf {
    pub num_dimensions: u32,
    pub num_clusters: u32,
    pub num_horizontal_dimensions: u32,
    pub num_vertical_dimensions: u32,
    pub clusters: Vec<PdxCluster>,
}

impl IndexPdxIvf {
    pub fn new(n_points: usize, d: usize) -> Self {
        let split = get_dimension_split(d);
        debug_assert_eq!(split.horizontal_d % H_DIM_SIZE, 0);

        let full_clusters = n_points / VECTOR_CHUNK_SIZE;
        let n_remaining = n_points % VECTOR_CHUNK_SIZE;
        let total_clusters = full_clusters + (if n_remaining > 0 { 1 } else { 0 });

        let mut clusters = Vec::with_capacity(total_clusters);
        let mut data_offset = 0usize;
        let mut indices_offset = 0usize;
        let mut aux_offset = 0usize;
        for ci in 0..full_clusters {
            clusters.push(PdxCluster {
                num_embeddings: VECTOR_CHUNK_SIZE as u32,
                data_offset,
                indices_offset,
                aux_offset,
            });
            data_offset += VECTOR_CHUNK_SIZE * d;
            indices_offset += VECTOR_CHUNK_SIZE;
            aux_offset += VECTOR_CHUNK_SIZE * split.vertical_d;
            let _ = ci;
        }
        if n_remaining > 0 {
            clusters.push(PdxCluster {
                num_embeddings: n_remaining as u32,
                data_offset,
                indices_offset,
                aux_offset,
            });
        }

        Self {
            num_dimensions: d as u32,
            num_clusters: total_clusters as u32,
            num_horizontal_dimensions: split.horizontal_d as u32,
            num_vertical_dimensions: split.vertical_d as u32,
            clusters,
        }
    }

    /// PDX data buffer length needed.
    pub fn data_buffer_len(&self) -> usize {
        let d = self.num_dimensions as usize;
        let n: usize = self.clusters.iter().map(|c| c.num_embeddings as usize).sum();
        // Same packed size as row-major: every element is stored exactly once
        // (all dimensions, all points).
        n * d
    }

    pub fn aux_buffer_len(&self) -> usize {
        let v = self.num_vertical_dimensions as usize;
        let n: usize = self.clusters.iter().map(|c| c.num_embeddings as usize).sum();
        n * v
    }

    /// Row indices buffer length (one `u32` per data point).
    pub fn indices_buffer_len(&self) -> usize {
        self.clusters.iter().map(|c| c.num_embeddings as usize).sum()
    }
}

/// Transform a row-major matrix to PDX layout.
///
/// Mirrors `PDXLayout::PDXify<FULLY_TRANSPOSED=false>`. Within each
/// `CHUNK_SIZE` block we write:
///   - vertical part: `vertical_d × CHUNK` dim-major (transpose of leftmost
///     `vertical_d` columns).
///   - then for each `H_DIM_SIZE`-wide horizontal block: `CHUNK × H_DIM_SIZE`
///     row-major (the corresponding columns of the input).
pub fn pdxify(in_vectors: &[f32], out_pdx: &mut [f32], n: usize, d: usize) {
    pdxify_with_chunk(in_vectors, out_pdx, n, d, VECTOR_CHUNK_SIZE);
}

pub fn pdxify_with_chunk(
    in_vectors: &[f32],
    out_pdx: &mut [f32],
    n: usize,
    d: usize,
    chunk_size: usize,
) {
    let split = get_dimension_split(d);
    let horizontal_d = split.horizontal_d;
    let vertical_d = split.vertical_d;
    debug_assert!(horizontal_d % H_DIM_SIZE == 0);
    debug_assert_eq!(in_vectors.len(), n * d);
    debug_assert_eq!(out_pdx.len(), n * d);

    let full_chunks = n / chunk_size;
    let n_remaining = n % chunk_size;

    for chunk in 0..full_chunks {
        let chunk_offset = chunk * chunk_size * d;
        let in_chunk = &in_vectors[chunk_offset..chunk_offset + chunk_size * d];
        let out_chunk = &mut out_pdx[chunk_offset..chunk_offset + chunk_size * d];
        write_chunk(in_chunk, out_chunk, chunk_size, d, vertical_d, horizontal_d);
    }
    if n_remaining > 0 {
        let chunk_offset = full_chunks * chunk_size * d;
        let in_chunk = &in_vectors[chunk_offset..chunk_offset + n_remaining * d];
        let out_chunk = &mut out_pdx[chunk_offset..chunk_offset + n_remaining * d];
        write_chunk(
            in_chunk,
            out_chunk,
            n_remaining,
            d,
            vertical_d,
            horizontal_d,
        );
    }
}

fn write_chunk(
    in_chunk: &[f32],
    out_chunk: &mut [f32],
    chunk_rows: usize,
    d: usize,
    vertical_d: usize,
    horizontal_d: usize,
) {
    // Vertical (dim-major) block: out[j * chunk_rows + i] = in[i * d + j] for j in 0..vertical_d.
    let (vert_out, rest) = out_chunk.split_at_mut(vertical_d * chunk_rows);
    for j in 0..vertical_d {
        for i in 0..chunk_rows {
            vert_out[j * chunk_rows + i] = in_chunk[i * d + j];
        }
    }

    // Horizontal blocks: each H_DIM_SIZE-wide column slice, row-major within.
    let mut h_off = 0usize;
    let mut block_start = 0usize;
    while h_off < horizontal_d {
        let h_block = &mut rest[block_start..block_start + chunk_rows * H_DIM_SIZE];
        for i in 0..chunk_rows {
            let src = vertical_d + h_off;
            let row = &in_chunk[i * d + src..i * d + src + H_DIM_SIZE];
            let dst = &mut h_block[i * H_DIM_SIZE..(i + 1) * H_DIM_SIZE];
            dst.copy_from_slice(row);
        }
        h_off += H_DIM_SIZE;
        block_start += chunk_rows * H_DIM_SIZE;
    }
}

/// Inverse of `pdxify`: reconstruct the row-major matrix. Used only by tests.
pub fn unpdxify(pdx: &[f32], out: &mut [f32], n: usize, d: usize) {
    unpdxify_with_chunk(pdx, out, n, d, VECTOR_CHUNK_SIZE);
}

pub fn unpdxify_with_chunk(
    pdx: &[f32],
    out: &mut [f32],
    n: usize,
    d: usize,
    chunk_size: usize,
) {
    let split = get_dimension_split(d);
    let vertical_d = split.vertical_d;
    let horizontal_d = split.horizontal_d;

    let full_chunks = n / chunk_size;
    let n_remaining = n % chunk_size;

    for chunk in 0..full_chunks {
        let chunk_offset = chunk * chunk_size * d;
        let in_chunk = &pdx[chunk_offset..chunk_offset + chunk_size * d];
        let out_chunk = &mut out[chunk_offset..chunk_offset + chunk_size * d];
        read_chunk(in_chunk, out_chunk, chunk_size, d, vertical_d, horizontal_d);
    }
    if n_remaining > 0 {
        let chunk_offset = full_chunks * chunk_size * d;
        let in_chunk = &pdx[chunk_offset..chunk_offset + n_remaining * d];
        let out_chunk = &mut out[chunk_offset..chunk_offset + n_remaining * d];
        read_chunk(
            in_chunk,
            out_chunk,
            n_remaining,
            d,
            vertical_d,
            horizontal_d,
        );
    }
}

fn read_chunk(
    in_chunk: &[f32],
    out_chunk: &mut [f32],
    chunk_rows: usize,
    d: usize,
    vertical_d: usize,
    horizontal_d: usize,
) {
    let (vert_in, rest) = in_chunk.split_at(vertical_d * chunk_rows);
    for j in 0..vertical_d {
        for i in 0..chunk_rows {
            out_chunk[i * d + j] = vert_in[j * chunk_rows + i];
        }
    }
    let mut h_off = 0usize;
    let mut block_start = 0usize;
    while h_off < horizontal_d {
        let h_block = &rest[block_start..block_start + chunk_rows * H_DIM_SIZE];
        for i in 0..chunk_rows {
            let dst_off = i * d + vertical_d + h_off;
            out_chunk[dst_off..dst_off + H_DIM_SIZE]
                .copy_from_slice(&h_block[i * H_DIM_SIZE..(i + 1) * H_DIM_SIZE]);
        }
        h_off += H_DIM_SIZE;
        block_start += chunk_rows * H_DIM_SIZE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_split_invariants() {
        for d in [4, 64, 128, 192, 256, 384, 512, 768, 1024, 1536] {
            let s = get_dimension_split(d);
            assert_eq!(s.horizontal_d + s.vertical_d, d, "d={d}");
            assert_eq!(s.horizontal_d % H_DIM_SIZE, 0, "d={d} horiz not multiple of H");
        }
    }

    #[test]
    fn pdxify_roundtrips() {
        let n = 5usize;
        let d = 192usize;
        let in_vec: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.5 - 7.0).collect();
        let mut pdx = vec![0.0; n * d];
        let mut back = vec![0.0; n * d];
        pdxify_with_chunk(&in_vec, &mut pdx, n, d, 4);
        unpdxify_with_chunk(&pdx, &mut back, n, d, 4);
        assert_eq!(in_vec, back);
    }

    #[test]
    fn index_buffer_lengths() {
        let n = 5_000usize;
        let d = 192usize;
        let idx = IndexPdxIvf::new(n, d);
        assert_eq!(idx.data_buffer_len(), n * d);
        assert_eq!(idx.indices_buffer_len(), n);
        let v = idx.num_vertical_dimensions as usize;
        assert_eq!(idx.aux_buffer_len(), n * v);
    }
}
