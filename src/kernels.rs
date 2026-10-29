//! Float kernels. The matrix-multiply micro-kernel is explicit NEON on aarch64; elsewhere it is
//! written so LLVM auto-vectorizes it (SIMD128 on wasm, SSE on x86-64).

use alloc::vec;
use alloc::vec::Vec;

#[cfg(not(feature = "std"))]
use crate::compat::F32Ext;

// Micro-kernel tile: MR rows x NR columns of accumulators. aarch64 has 32 vector registers
// (4x16 = 16 accumulators); wasm SIMD / x86-64 SSE have 16 (2x16 = 8). Measured, see README.
const NR: usize = 16; // output columns per panel
#[cfg(target_arch = "aarch64")]
const MR: usize = 4; // output rows per micro-kernel call
#[cfg(not(target_arch = "aarch64"))]
const MR: usize = 2;
const MB: usize = 64; // rows per cache block

/// Right-hand matrix packed into panels of NR output columns, k-major inside a panel, so the
/// micro-kernel reads NR contiguous weights per k step (outer-product form, no reductions).
pub struct Packed {
    pub n: usize,
    pub k: usize,
    data: Vec<f32>, // ceil(n / NR) panels x k x NR, zero-padded
}

impl Packed {
    /// From b (n x k): row j holds the weights of output column j (PyTorch Linear layout).
    pub fn from_rows(b: &[f32], n: usize, k: usize) -> Packed {
        let mut data = vec![0.0; n.div_ceil(NR) * k * NR];
        for j in 0..n {
            let (panel, l) = (j / NR, j % NR);
            for (p, v) in b[j * k..(j + 1) * k].iter().enumerate() {
                data[(panel * k + p) * NR + l] = *v;
            }
        }
        Packed { n, k, data }
    }

    /// From b (k x n), row-major.
    pub fn from_cols(b: &[f32], k: usize, n: usize) -> Packed {
        let mut data = vec![0.0; n.div_ceil(NR) * k * NR];
        for (p, row) in b.chunks_exact(n).enumerate() {
            for (j, v) in row.iter().enumerate() {
                data[((j / NR) * k + p) * NR + j % NR] = *v;
            }
        }
        Packed { n, k, data }
    }

    fn panel(&self, i: usize) -> &[f32] {
        &self.data[i * self.k * NR..(i + 1) * self.k * NR]
    }
}

/// out (m x n) = a (m x k) * B (+ bias per column).
pub fn gemm(a: &[f32], m: usize, b: &Packed, bias: Option<&[f32]>) -> Vec<f32> {
    let (n, k) = (b.n, b.k);
    assert!(a.len() >= m * k);
    let mut out = vec![0.0; m * n];
    #[cfg(all(feature = "threads", not(target_arch = "wasm32")))]
    {
        let threads = threads_for(m * n * k);
        if threads > 1 && m >= 2 * MB {
            let per = m.div_ceil(threads).next_multiple_of(MR);
            std::thread::scope(|s| {
                for (a, out) in a[..m * k].chunks(per * k).zip(out.chunks_mut(per * n)) {
                    s.spawn(move || gemm_rows(a, b, bias, out));
                }
            });
            return out;
        }
    }
    gemm_rows(&a[..m * k], b, bias, &mut out);
    out
}

fn gemm_rows(a: &[f32], b: &Packed, bias: Option<&[f32]>, out: &mut [f32]) {
    let (n, k) = (b.n, b.k);
    let m = a.len() / k;
    let mut tile = [[0f32; NR]; MR];
    for i0 in (0..m).step_by(MB) {
        let i1 = (i0 + MB).min(m);
        for jp in 0..n.div_ceil(NR) {
            let panel = b.panel(jp);
            let j0 = jp * NR;
            let cols = NR.min(n - j0);
            let mut init = [0f32; NR];
            if let Some(bias) = bias {
                init[..cols].copy_from_slice(&bias[j0..j0 + cols]);
            }
            let mut i = i0;
            while i < i1 {
                let rows = MR.min(i1 - i);
                if rows == MR {
                    kernel::<MR>(&a[i * k..(i + MR) * k], k, panel, &init, &mut tile);
                } else {
                    for r in 0..rows {
                        let mut one = [[0f32; NR]; 1];
                        kernel::<1>(&a[(i + r) * k..(i + r + 1) * k], k, panel, &init, &mut one);
                        tile[r] = one[0];
                    }
                }
                for r in 0..rows {
                    out[(i + r) * n + j0..(i + r) * n + j0 + cols].copy_from_slice(&tile[r][..cols]);
                }
                i += rows;
            }
        }
    }
}

/// out = init + rows(a) x panel, for R rows of `a` (each k long) and one panel of NR columns.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn kernel<const R: usize>(a: &[f32], k: usize, panel: &[f32], init: &[f32; NR], out: &mut [[f32; NR]; R]) {
    let rows: [&[f32]; R] = core::array::from_fn(|r| &a[r * k..(r + 1) * k]);
    let mut acc = [*init; R];
    for (p, b) in panel[..k * NR].chunks_exact(NR).enumerate() {
        for r in 0..R {
            let x = rows[r][p];
            for l in 0..NR {
                acc[r][l] += x * b[l];
            }
        }
    }
    *out = acc;
}

/// NEON version: R x 16 accumulators in q registers, fused multiply-add (fmla by element).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code)]
fn kernel<const R: usize>(a: &[f32], k: usize, panel: &[f32], init: &[f32; NR], out: &mut [[f32; NR]; R]) {
    use core::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
    const { assert!(NR == 16) };
    assert!(a.len() >= R * k && panel.len() >= k * NR);
    // SAFETY: NEON is mandatory on aarch64; every load/store stays inside the slices checked above
    // (row r reads a[r*k + p], p < k; panel reads panel[p*16 .. p*16+16], p < k).
    unsafe {
        let ip = init.as_ptr();
        let start: [float32x4_t; 4] = [vld1q_f32(ip), vld1q_f32(ip.add(4)), vld1q_f32(ip.add(8)), vld1q_f32(ip.add(12))];
        let mut acc = [start; R];
        let (ap, bp) = (a.as_ptr(), panel.as_ptr());
        for p in 0..k {
            let b = bp.add(p * NR);
            let (b0, b1, b2, b3) = (vld1q_f32(b), vld1q_f32(b.add(4)), vld1q_f32(b.add(8)), vld1q_f32(b.add(12)));
            for (r, acc) in acc.iter_mut().enumerate() {
                let x = vdupq_n_f32(*ap.add(r * k + p));
                acc[0] = vfmaq_f32(acc[0], b0, x);
                acc[1] = vfmaq_f32(acc[1], b1, x);
                acc[2] = vfmaq_f32(acc[2], b2, x);
                acc[3] = vfmaq_f32(acc[3], b3, x);
            }
        }
        for (o, acc) in out.iter_mut().zip(&acc) {
            for (j, v) in acc.iter().enumerate() {
                vst1q_f32(o.as_mut_ptr().add(4 * j), *v);
            }
        }
    }
}

#[cfg(all(feature = "threads", not(target_arch = "wasm32")))]
fn threads_for(work: usize) -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    if work < 4_000_000 {
        return 1;
    }
    *N.get_or_init(|| {
        let env = std::env::var("TINYLID_THREADS").ok().and_then(|v| v.parse().ok());
        env.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get().min(8)))
    })
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len().min(b.len());
    let k8 = k / 8 * 8;
    let (x, y) = (&a[..k8], &b[..k8]);
    let mut acc = [0f32; 8];
    for c in 0..k8 / 8 {
        for l in 0..8 {
            acc[l] += x[c * 8 + l] * y[c * 8 + l];
        }
    }
    acc.iter().sum::<f32>() + (k8..k).map(|p| a[p] * b[p]).sum::<f32>()
}

/// Exact GELU, 0.5 x (1 + erf(x / sqrt 2)), with erf from Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7).
pub fn gelu(x: f32) -> f32 {
    let z = x * core::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
    let poly = t * (0.254_829_6 + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let erf = 1.0 - poly * exp(-z * z);
    0.5 * x * (1.0 + if z < 0.0 { -erf } else { erf })
}

/// exp for x <= ~88 via 2^x = 2^i * 2^f with a degree-5 polynomial (rel. err ~2e-7).
/// Branch-free so loops over it vectorize, unlike the libm call.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let t = (x * core::f32::consts::LOG2_E).clamp(-126.0, 126.0);
    let i = t.floor();
    let f = t - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355))));
    f32::from_bits(((i as i32 + 127) as u32) << 23) * p
}

/// LayerNorm (eps 1e-5) over each row of length d, in place.
pub fn layer_norm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    for r in x.chunks_exact_mut(d) {
        let mean = r.iter().sum::<f32>() / d as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for ((v, w), b) in r.iter_mut().zip(w).zip(b) {
            *v = (*v - mean) * inv * w + b;
        }
    }
}

pub fn softmax(x: &mut [f32]) {
    // 8 independent lanes so the reductions vectorize (float max/sum are not reassociated otherwise)
    let mut m = [f32::MIN; 8];
    for c in x.chunks(8) {
        for (m, v) in m.iter_mut().zip(c) {
            *m = m.max(*v);
        }
    }
    let max = m.iter().copied().fold(f32::MIN, f32::max);
    x.iter_mut().for_each(|v| *v = exp(*v - max));
    let mut s = [0f32; 8];
    for c in x.chunks(8) {
        for (s, v) in s.iter_mut().zip(c) {
            *s += v;
        }
    }
    let inv = 1.0 / s.iter().sum::<f32>();
    x.iter_mut().for_each(|v| *v *= inv);
}
