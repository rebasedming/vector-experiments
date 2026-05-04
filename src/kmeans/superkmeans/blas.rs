//! BLAS dispatch.
//!
//! - **macOS**: Apple Accelerate (auto-dispatches AMX on Apple Silicon) via
//!   `accelerate-src` linking + manual `cblas_sgemm` FFI.
//! - **Linux aarch64**: system OpenBLAS (apt: `libopenblas-dev`) via
//!   `openblas-src` linking + the same `cblas_sgemm` FFI. On Graviton4 /
//!   Neoverse-V2 this gets us SVE2 sgemm kernels.
//! - **Other targets**: pure-Rust `matrixmultiply::sgemm`. Cross-platform
//!   but NEON/AVX-only, no SVE2 / AMX.

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

#[derive(Clone, Copy)]
pub enum Trans {
    No,
    Yes,
}

/// Row-major sgemm: `C[m,n] = α · op(A) · op(B) + β · C` where each matrix is
/// row-major and `op` is governed by `trans_a`/`trans_b`.
///
/// `lda`/`ldb`/`ldc` are row strides (i.e. number of columns in the
/// underlying storage, ignoring the transpose). This matches the C++
/// `BlasMatrixMultiplication` semantics.
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
        // Map (trans_a, trans_b, lda, ldb) into matrixmultiply's (rsa,csa,rsb,csb).
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
