use rayon::prelude::*;

pub use blas::{init as blas_init, sgemm_row_major, Trans};

#[cfg(target_os = "macos")]
pub const X_BATCH_SIZE: usize = 40_960;
#[cfg(target_os = "macos")]
pub const Y_BATCH_SIZE: usize = 2_048;

#[cfg(not(target_os = "macos"))]
pub const X_BATCH_SIZE: usize = 4_096;
#[cfg(not(target_os = "macos"))]
pub const Y_BATCH_SIZE: usize = 1_024;

const MINI_BATCH_SIZE: usize = 256;

#[inline]
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        return squared_l2_neon(a, b);
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum
    }
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn squared_l2_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let n = a.len();
    let mut i = 0usize;
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    while i + 4 <= n {
        unsafe {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            let dv = vsubq_f32(av, bv);
            sum_vec = vfmaq_f32(sum_vec, dv, dv);
        }
        i += 4;
    }
    let mut sum = unsafe { vaddvq_f32(sum_vec) };
    while i < n {
        let diff = a[i] - b[i];
        sum += diff * diff;
        i += 1;
    }
    sum
}

fn blas_matmul(
    batch_x: &[f32],
    batch_y: &[f32],
    batch_n_x: usize,
    batch_n_y: usize,
    d: usize,
    tmp_distances: &mut [f32],
) {
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
                    d,
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
            d,
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

            blas_matmul(batch_x, batch_y, batch_n_x, batch_n_y, d, tmp);

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

mod blas {
    #[cfg(any(target_os = "macos", all(target_os = "linux", target_arch = "aarch64")))]
    mod cblas {
        pub const CBLAS_ROW_MAJOR: i32 = 101;
        pub const CBLAS_NO_TRANS: i32 = 111;
        pub const CBLAS_TRANS: i32 = 112;

        extern "C" {
            pub fn cblas_sgemm(
                order: i32,
                trans_a: i32,
                trans_b: i32,
                m: i32,
                n: i32,
                k: i32,
                alpha: f32,
                a: *const f32,
                lda: i32,
                b: *const f32,
                ldb: i32,
                beta: f32,
                c: *mut f32,
                ldc: i32,
            );
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    mod blis_threads {
        extern "C" {
            fn bli_thread_set_num_threads(n: i64);
        }

        pub fn init() {
            let n = std::thread::available_parallelism()
                .map(|x| x.get() as i64)
                .unwrap_or(1);
            unsafe { bli_thread_set_num_threads(n) };
        }
    }

    pub fn init() {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        blis_threads::init();
    }

    #[derive(Clone, Copy)]
    pub enum Trans {
        No,
        Yes,
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_row_major(
        trans_a: Trans,
        trans_b: Trans,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        debug_assert!(c.len() >= m * ldc);

        #[cfg(any(target_os = "macos", all(target_os = "linux", target_arch = "aarch64")))]
        {
            use cblas::*;
            let cblas_trans_a = match trans_a {
                Trans::No => CBLAS_NO_TRANS,
                Trans::Yes => CBLAS_TRANS,
            };
            let cblas_trans_b = match trans_b {
                Trans::No => CBLAS_NO_TRANS,
                Trans::Yes => CBLAS_TRANS,
            };
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    cblas_trans_a,
                    cblas_trans_b,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    a.as_ptr(),
                    lda as i32,
                    b.as_ptr(),
                    ldb as i32,
                    beta,
                    c.as_mut_ptr(),
                    ldc as i32,
                );
            }
            return;
        }

        #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_arch = "aarch64"))))]
        {
            let (rsa, csa) = match trans_a {
                Trans::No => (lda as isize, 1isize),
                Trans::Yes => (1isize, lda as isize),
            };
            let (rsb, csb) = match trans_b {
                Trans::No => (ldb as isize, 1isize),
                Trans::Yes => (1isize, ldb as isize),
            };
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n,
                    alpha,
                    a.as_ptr(),
                    rsa,
                    csa,
                    b.as_ptr(),
                    rsb,
                    csb,
                    beta,
                    c.as_mut_ptr(),
                    ldc as isize,
                    1,
                );
            }
        }
    }
}
