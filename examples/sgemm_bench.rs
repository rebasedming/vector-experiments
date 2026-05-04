// Standalone sgemm throughput probe matching the ASSIGN workload shape.
// Same dispatch logic as the experiment's blas.rs: Accelerate on macOS,
// matrixmultiply (NEON, no SVE2) elsewhere.

use std::time::Instant;

#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
extern crate openblas_src;

#[cfg(any(target_os = "macos", all(target_os = "linux", target_arch = "aarch64")))]
mod cblas {
    pub const CBLAS_ROW_MAJOR: i32 = 101;
    pub const CBLAS_NO_TRANS: i32 = 111;
    pub const CBLAS_TRANS: i32 = 112;
    extern "C" {
        pub fn cblas_sgemm(
            order: i32, ta: i32, tb: i32,
            m: i32, n: i32, k: i32,
            alpha: f32,
            a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: f32,
            c: *mut f32, ldc: i32,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn sgemm_x_yT(n: usize, k: usize, d: usize, x: &[f32], y: &[f32], tmp: &mut [f32]) {
    // tmp[N,K] = X[N,D] * Y^T[D,K] (Y given as row-major K x D).
    #[cfg(any(target_os = "macos", all(target_os = "linux", target_arch = "aarch64")))]
    unsafe {
        cblas::cblas_sgemm(
            cblas::CBLAS_ROW_MAJOR,
            cblas::CBLAS_NO_TRANS,
            cblas::CBLAS_TRANS,
            n as i32, k as i32, d as i32,
            1.0,
            x.as_ptr(), d as i32,
            y.as_ptr(), d as i32,
            0.0,
            tmp.as_mut_ptr(), k as i32,
        );
        return;
    }
    #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_arch = "aarch64"))))]
    unsafe {
        matrixmultiply::sgemm(
            n, d, k,
            1.0,
            x.as_ptr(), d as isize, 1,
            y.as_ptr(), 1, d as isize,
            0.0,
            tmp.as_mut_ptr(), k as isize, 1,
        );
    }
}

fn run(label: &str, n: usize, k: usize, d: usize) {
    println!("\n=== {label} (N={n}, K={k}, D={d}) ===");
    let flops = 2u64 * n as u64 * k as u64 * d as u64;
    println!("FLOPs: {} GFLOPs", flops / 1_000_000_000);

    let t = Instant::now();
    let x: Vec<f32> = vec![0.5; n * d];
    let y: Vec<f32> = vec![0.5; k * d];
    let mut tmp: Vec<f32> = vec![0.0; n * k];
    let alloc_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("alloc:    {alloc_ms:>7.0} ms");

    sgemm_x_yT(64, 64, d, &x[..64*d], &y[..64*d], &mut tmp[..64*64]);

    let mut best_ms = f64::MAX;
    for trial in 0..3 {
        let t = Instant::now();
        sgemm_x_yT(n, k, d, &x, &y, &mut tmp);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let gflops = flops as f64 / 1e9 / (ms / 1000.0);
        println!("trial {trial}: {ms:>7.0} ms  ({gflops:>6.1} GFLOPS)");
        best_ms = best_ms.min(ms);
    }
    let best_gflops = flops as f64 / 1e9 / (best_ms / 1000.0);
    println!("best:     {best_ms:>7.0} ms  ({best_gflops:>6.1} GFLOPS)");
}

fn main() {
    let backend = if cfg!(target_os = "macos") {
        "Apple Accelerate (auto-AMX)"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "OpenBLAS (system, SVE2 kernels)"
    } else {
        "matrixmultiply (pure-Rust NEON/AVX)"
    };
    println!("Backend: {backend}");

    run("ASSIGN-shape", 1_000_000, 2_048, 768);
    run("Cache-resident", 4096, 2048, 768);
}
