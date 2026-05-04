//! Distance / utility kernels.
//!
//! Mirrors `scalar_computers.h` and `neon_computers.h`. We dispatch by
//! target_arch so the binary on Apple Silicon and Linux ARM64 uses the NEON
//! kernels exactly as the C++ does, while x86_64 and other targets use the
//! scalar fallback (the C++ has AVX2/AVX-512 paths which we omit for now).
//!
//! Note: the C++ `neon_computers.h` deliberately falls back to a scalar
//! autovectorized loop for `L2 / f32 / Horizontal` on Apple specifically —
//! Apple's clang autovectorizer matches hand-written intrinsics there. We do
//! the same: rely on rustc's autovec for the L2 kernel on Apple, hand-write
//! NEON only for non-Apple ARM64. Sign-flip and the position-array compactor
//! always use NEON when available.

#[inline]
pub fn squared_l2_horizontal(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        return squared_l2_horizontal_neon(a, b);
    }
    #[allow(unreachable_code)]
    {
        squared_l2_horizontal_scalar(a, b)
    }
}

#[inline]
fn squared_l2_horizontal_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn squared_l2_horizontal_neon(a: &[f32], b: &[f32]) -> f32 {
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

/// `out[j] = data[j] ^ masks[j]` reinterpreted via `f32 <-> u32` bits.
/// NEON-vectorized on aarch64; scalar elsewhere.
pub fn flip_sign(data: &[f32], out: &mut [f32], masks: &[u32]) {
    debug_assert_eq!(data.len(), out.len());
    debug_assert_eq!(data.len(), masks.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let n = data.len();
        let mut j = 0usize;
        while j + 4 <= n {
            let v = vld1q_f32(data.as_ptr().add(j));
            let m = vld1q_u32(masks.as_ptr().add(j));
            let xored = veorq_u32(vreinterpretq_u32_f32(v), m);
            vst1q_f32(out.as_mut_ptr().add(j), vreinterpretq_f32_u32(xored));
            j += 4;
        }
        while j < n {
            let bits = data[j].to_bits() ^ masks[j];
            out[j] = f32::from_bits(bits);
            j += 1;
        }
        return;
    }

    #[allow(unreachable_code)]
    for ((d, o), m) in data.iter().zip(out.iter_mut()).zip(masks.iter()) {
        let bits = d.to_bits() ^ *m;
        *o = f32::from_bits(bits);
    }
}

/// Compact the indices `0..n_vectors` whose `pruning_distances[i] < threshold`.
/// NEON-accelerated on aarch64 (matches `SIMDUtilsComputer::InitPositionsArray`).
pub fn init_positions_array(
    pruning_positions: &mut [u32],
    pruning_distances: &[f32],
    pruning_threshold: f32,
    n_vectors: usize,
) -> usize {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        const W: usize = 4;
        let n_simd = (n_vectors / W) * W;
        let mut n_kept = 0usize;
        let threshold_vec = vdupq_n_f32(pruning_threshold);
        let mut i = 0usize;
        while i < n_simd {
            let dv = vld1q_f32(pruning_distances.as_ptr().add(i));
            let cmp = vcltq_f32(dv, threshold_vec);
            let any = vmaxvq_u32(cmp);
            if any != 0 {
                let mut mask = [0u32; W];
                vst1q_u32(mask.as_mut_ptr(), cmp);
                for k in 0..W {
                    pruning_positions[n_kept] = (i + k) as u32;
                    n_kept += (mask[k] != 0) as usize;
                }
            }
            i += W;
        }
        while i < n_vectors {
            pruning_positions[n_kept] = i as u32;
            if pruning_distances[i] < pruning_threshold {
                n_kept += 1;
            }
            i += 1;
        }
        return n_kept;
    }

    #[allow(unreachable_code)]
    {
        let mut n_kept = 0usize;
        for i in 0..n_vectors {
            pruning_positions[n_kept] = i as u32;
            if pruning_distances[i] < pruning_threshold {
                n_kept += 1;
            }
        }
        n_kept
    }
}
