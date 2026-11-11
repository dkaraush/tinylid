//! Spoken language identification with no dependencies.
//!
//! Model: Whisper-tiny encoder (4 layers, d=384) + attention pooling + linear head over 125
//! languages, weights stored as 4-bit integers with one fp16 scale per 32 weights (~4.3 MB file).
//! Weights are dequantized to f32 on load (~31 MB RAM).
//!
//! `no_std` + `alloc` without the default `std` feature; no I/O, no internal threads unless the
//! `threads` feature is on.
//!
//! ```no_run
//! let model = tinylid::Model::from_bytes(&std::fs::read("tinylid.bin").unwrap()).unwrap();
//! let audio_16k: Vec<f32> = vec![0.0; 16000];
//! let probs = model.predict(&audio_16k, tinylid::Mode::Fast);
//! for (lang, p) in model.top(&probs, 3) { println!("{lang} {p:.3}"); }
//! ```

#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![deny(unsafe_code)]

extern crate alloc;

mod compat;
#[cfg(any(feature = "ffi", feature = "wasm"))]
mod ffi;
mod kernels;
#[cfg(all(not(feature = "std"), feature = "c-api", not(target_arch = "wasm32"), not(test)))]
mod no_std_runtime;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;

#[cfg(not(feature = "std"))]
use compat::{F32Ext, F64Ext};
use kernels::{Packed, gelu, gemm, layer_norm, softmax};

/// Sample rate `predict` expects.

pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_BINS: usize = N_FFT / 2 + 1;
const MAX_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
const MAX_POS: usize = MAX_SAMPLES / HOP / 2; // 1500 encoder frames (50 per second)

/// How the clip is fed to the encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Only the real audio (rounded up to 20 ms). Cost grows with clip length; a 5 s clip is
    /// ~8x cheaper than `Exact`. Accuracy is 0-2.5 points lower because training padded to 30 s.
    Fast,
    /// Zero-pad to 30 s exactly as in training. Fixed cost (~18 GMAC).
    Exact,
}

struct Layer {
    ln1: (Vec<f32>, Vec<f32>),
    q: Packed,
    q_b: Vec<f32>,
    k: Packed,
    v: Packed,
    v_b: Vec<f32>,
    o: Packed,
    o_b: Vec<f32>,
    ln2: (Vec<f32>, Vec<f32>),
    fc1: Packed,
    fc1_b: Vec<f32>,
    fc2: Packed,
    fc2_b: Vec<f32>,
}

/// A loaded model. Immutable after loading, so one instance can serve concurrent `predict` calls.
pub struct Model {
    n_mels: usize,
    d: usize,
    heads: usize,
    langs: Vec<String>,
    mel_filters: Packed, // n_mels x N_BINS
    dft: Packed,         // (2 * N_BINS) x N_FFT: Hann-windowed cos/sin rows
    pos: Vec<f32>,       // MAX_POS x d sinusoids
    conv1: Packed,
    conv1_b: Vec<f32>,
    conv2: Packed,
    conv2_b: Vec<f32>,
    layers: Vec<Layer>,
    ln_out: (Vec<f32>, Vec<f32>),
    pool: Vec<f32>,
    pool_b: Vec<f32>,
    norm: (Vec<f32>, Vec<f32>),
    head: Packed,
    head_b: Vec<f32>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len()).ok_or("model file truncated")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, &'static str> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<usize, &'static str> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, &'static str> {
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// A tensor as written by export_rust.py: f32, or 4-bit with an fp16 scale per `group` weights.
    fn mat(&mut self, rows: usize, cols: usize) -> Result<Vec<f32>, &'static str> {
        let kind = self.u8()?;
        if self.u32()? != rows || self.u32()? != cols {
            return Err("unexpected tensor shape");
        }
        let n = rows * cols;
        let data = match kind {
            0 => self.f32s(n)?,
            1 => {
                let group = self.u32()?;
                if group == 0 || n % group != 0 || n % 2 != 0 {
                    return Err("bad quantization group");
                }
                let scales: Vec<f32> = self
                    .take(n / group * 2)?
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let packed = self.take(n / 2)?;
                (0..n).map(|i| (((packed[i / 2] >> (4 * (i % 2))) & 15) as f32 - 8.0) * scales[i / group]).collect()
            }
            _ => return Err("unknown tensor kind"),
        };
        Ok(data)
    }

    fn packed(&mut self, rows: usize, cols: usize) -> Result<Packed, &'static str> {
        Ok(Packed::from_rows(&self.mat(rows, cols)?, rows, cols))
    }

    fn vec(&mut self, n: usize) -> Result<Vec<f32>, &'static str> {
        self.mat(n, 1)
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal: renormalize
            let (mut e, mut m) = (127 - 15 + 1, man);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

impl Model {
    /// Parse a model file produced by `export_rust.py`.
    pub fn from_bytes(buf: &[u8]) -> Result<Model, &'static str> {
        let mut r = Reader { buf, pos: 0 };
        if r.take(4)? != b"TLID" || r.u32()? != 1 {
            return Err("not a tinylid v1 model file");
        }
        let (n_mels, d, heads, n_layers, n_langs) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        if heads == 0 || d % heads != 0 || d % 2 != 0 || n_layers > 64 {
            return Err("bad model header");
        }
        let ffn = (0..n_layers).map(|_| r.u32()).collect::<Result<Vec<_>, _>>()?;
        let mut langs = Vec::with_capacity(n_langs);
        for _ in 0..n_langs {
            let len = r.u8()? as usize;
            langs.push(String::from_utf8(r.take(len)?.to_vec()).map_err(|_| "bad language code")?);
        }
        let mel_filters = Packed::from_rows(&r.f32s(n_mels * N_BINS)?, n_mels, N_BINS);
        let conv1 = r.packed(d, n_mels * 3)?;
        let conv1_b = r.vec(d)?;
        let conv2 = r.packed(d, d * 3)?;
        let conv2_b = r.vec(d)?;
        let mut layers = Vec::with_capacity(n_layers);
        for &f in &ffn {
            layers.push(Layer {
                ln1: (r.vec(d)?, r.vec(d)?),
                q: r.packed(d, d)?,
                q_b: r.vec(d)?,
                k: r.packed(d, d)?,
                v: r.packed(d, d)?,
                v_b: r.vec(d)?,
                o: r.packed(d, d)?,
                o_b: r.vec(d)?,
                ln2: (r.vec(d)?, r.vec(d)?),
                fc1: r.packed(f, d)?,
                fc1_b: r.vec(f)?,
                fc2: r.packed(d, f)?,
                fc2_b: r.vec(d)?,
            });
        }
        let ln_out = (r.vec(d)?, r.vec(d)?);
        let pool = r.mat(1, d)?;
        let pool_b = r.vec(1)?;
        let norm = (r.vec(d)?, r.vec(d)?);
        let head = r.packed(n_langs, d)?;
        let head_b = r.vec(n_langs)?;
        if r.pos != buf.len() {
            return Err("trailing bytes in model file");
        }
        Ok(Model {
            n_mels,
            d,
            heads,
            langs,
            mel_filters,
            dft: Packed::from_rows(&dft_basis(), 2 * N_BINS, N_FFT),
            pos: sinusoids(MAX_POS, d),
            conv1,
            conv1_b,
            conv2,
            conv2_b,
            layers,
            ln_out,
            pool,
            pool_b,
            norm,
            head,
            head_b,
        })
    }

    /// Language codes (ISO 639-1 where one exists), in the order of `predict`'s output.
    pub fn languages(&self) -> &[String] {
        &self.langs
    }

    /// Language probabilities for mono 16 kHz audio in [-1, 1]. Only the first 30 s are used.
    pub fn predict(&self, audio: &[f32], mode: Mode) -> Vec<f32> {
        let n = audio.len().min(MAX_SAMPLES);
        // Whole encoder frames (2 mel hops = 20 ms); at least one.
        let padded = match mode {
            Mode::Exact => MAX_SAMPLES,
            Mode::Fast => n.div_ceil(2 * HOP).max(1) * 2 * HOP,
        };
        let mut x = vec![0.0; padded];
        x[..n].copy_from_slice(&audio[..n]);
        let mel = self.log_mel(&x);
        let valid = (n / (2 * HOP)).clamp(1, padded / (2 * HOP));
        self.classify(&mel, padded / HOP, valid)
    }

    /// Like `predict` for audio at any sample rate (resampled to 16 kHz first).
    pub fn predict_resampled(&self, audio: &[f32], sample_rate: u32, mode: Mode) -> Vec<f32> {
        if sample_rate == SAMPLE_RATE {
            self.predict(audio, mode)
        } else {
            self.predict(&resample(audio, sample_rate, SAMPLE_RATE), mode)
        }
    }

    /// Log-mel of the first 30 s of 16 kHz audio, zero-padded, channel-major (n_mels x 3000):
    /// the input of Whisper-style models such as Ear.
    pub fn log_mel_30s(&self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len().min(MAX_SAMPLES);
        let mut x = vec![0.0; MAX_SAMPLES];
        x[..n].copy_from_slice(&audio[..n]);
        let mel = self.log_mel(&x);
        let frames = MAX_SAMPLES / HOP;
        let mut out = vec![0.0; self.n_mels * frames];
        for (t, row) in mel.chunks_exact(self.n_mels).enumerate() {
            for (m, v) in row.iter().enumerate() {
                out[m * frames + t] = *v;
            }
        }
        out
    }

    /// The `n` most likely languages, best first.
    pub fn top<'a>(&'a self, probs: &[f32], n: usize) -> Vec<(&'a str, f32)> {
        let mut idx: Vec<usize> = (0..probs.len().min(self.langs.len())).collect();
        idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        idx.into_iter().take(n).map(|i| (self.langs[i].as_str(), probs[i])).collect()
    }

    /// Whisper log-mel: (frames x n_mels), frames = audio.len() / HOP.
    fn log_mel(&self, audio: &[f32]) -> Vec<f32> {
        let frames = audio.len() / HOP;
        let last = audio.len() as isize - 1;
        let mut windows = vec![0.0; frames * N_FFT];
        for (t, w) in windows.chunks_exact_mut(N_FFT).enumerate() {
            for (j, v) in w.iter_mut().enumerate() {
                // centered frames with reflect padding (torch.stft center=True)
                let mut i = (t * HOP + j) as isize - (N_FFT / 2) as isize;
                if i < 0 {
                    i = -i;
                }
                if i > last {
                    i = 2 * last - i;
                }
                *v = audio[i.clamp(0, last) as usize];
            }
        }
        let spec = gemm(&windows, frames, &self.dft, None);
        let power: Vec<f32> = spec.chunks_exact(2).map(|c| c[0] * c[0] + c[1] * c[1]).collect();
        let mut mel = gemm(&power, frames, &self.mel_filters, None);
        let mut max = f32::MIN;
        for v in mel.iter_mut() {
            *v = v.max(1e-10).log10();
            max = max.max(*v);
        }
        for v in mel.iter_mut() {
            *v = (v.max(max - 8.0) + 4.0) / 4.0;
        }
        mel
    }

    fn classify(&self, mel: &[f32], frames: usize, valid: usize) -> Vec<f32> {
        let d = self.d;
        // conv1 (k=3, pad 1) + GELU, as a matmul over [mel channel][tap] columns
        let col = im2col(mel, frames, self.n_mels, frames, 1);
        let mut h = gemm(&col, frames, &self.conv1, Some(&self.conv1_b));
        h.iter_mut().for_each(|v| *v = gelu(*v));
        // conv2 (k=3, stride 2, pad 1) + GELU + positions
        let t = frames / 2;
        let col = im2col(&h, frames, d, t, 2);
        let mut x = gemm(&col, t, &self.conv2, Some(&self.conv2_b));
        for (v, p) in x.iter_mut().zip(&self.pos) {
            *v = gelu(*v) + p;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            // pooling reads only the first `valid` frames: the last layer computes just those rows
            let rows = if i + 1 == self.layers.len() { valid } else { t };
            x = self.layer(&x, t, rows, layer);
        }
        layer_norm(&mut x, d, &self.ln_out.0, &self.ln_out.1);

        // attention pooling over the frames that saw real audio
        let mut score: Vec<f32> =
            x.chunks_exact(d).take(valid).map(|r| kernels::dot(r, &self.pool) + self.pool_b[0]).collect();
        softmax(&mut score);
        let mut pooled = vec![0.0; d];
        for (r, w) in x.chunks_exact(d).zip(&score) {
            pooled.iter_mut().zip(r).for_each(|(p, v)| *p += w * v);
        }
        layer_norm(&mut pooled, d, &self.norm.0, &self.norm.1);
        let mut probs = gemm(&pooled, 1, &self.head, Some(&self.head_b));
        softmax(&mut probs);
        probs
    }

    /// Pre-LN transformer layer (Whisper encoder) over x (t x d). Returns the first `rows` output
    /// rows only (all t rows still serve as keys/values).
    fn layer(&self, x: &[f32], t: usize, rows: usize, l: &Layer) -> Vec<f32> {
        let (d, heads) = (self.d, self.heads);
        let dh = d / heads;
        let mut h = x.to_vec();
        layer_norm(&mut h, d, &l.ln1.0, &l.ln1.1);
        let mut q = gemm(&h, rows, &l.q, Some(&l.q_b));
        let scale = 1.0 / (dh as f32).sqrt();
        q.iter_mut().for_each(|v| *v *= scale);
        let k = gemm(&h, t, &l.k, None);
        let v = gemm(&h, t, &l.v, Some(&l.v_b));

        let mut ctx = vec![0.0; rows * d];
        let (mut qh, mut kh, mut vh) = (vec![0.0; rows * dh], vec![0.0; t * dh], vec![0.0; t * dh]);
        for hd in 0..heads {
            let cols = hd * dh..(hd + 1) * dh;
            for i in 0..t {
                kh[i * dh..(i + 1) * dh].copy_from_slice(&k[i * d..][cols.clone()]);
                vh[i * dh..(i + 1) * dh].copy_from_slice(&v[i * d..][cols.clone()]);
            }
            for i in 0..rows {
                qh[i * dh..(i + 1) * dh].copy_from_slice(&q[i * d..][cols.clone()]);
            }
            let mut att = gemm(&qh, rows, &Packed::from_rows(&kh, t, dh), None);
            att.chunks_exact_mut(t).for_each(softmax);
            let out = gemm(&att, rows, &Packed::from_cols(&vh, t, dh), None);
            for i in 0..rows {
                ctx[i * d..][cols.clone()].copy_from_slice(&out[i * dh..(i + 1) * dh]);
            }
        }
        let mut x: Vec<f32> = x[..rows * d].to_vec();
        let o = gemm(&ctx, rows, &l.o, Some(&l.o_b));
        x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

        let mut h = x.clone();
        layer_norm(&mut h, d, &l.ln2.0, &l.ln2.1);
        let mut f = gemm(&h, rows, &l.fc1, Some(&l.fc1_b));
        f.iter_mut().for_each(|v| *v = gelu(*v));
        let o = gemm(&f, rows, &l.fc2, Some(&l.fc2_b));
        x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
        x
    }
}

/// Rows of [channel][tap] for a k=3, pad=1 convolution over time-major input (frames x ch).
fn im2col(x: &[f32], frames: usize, ch: usize, out_frames: usize, stride: usize) -> Vec<f32> {
    let mut col = vec![0.0; out_frames * ch * 3];
    for (t, row) in col.chunks_exact_mut(ch * 3).enumerate() {
        for j in 0..3 {
            let src = (t * stride + j) as isize - 1;
            if src < 0 || src as usize >= frames {
                continue;
            }
            let s = &x[src as usize * ch..(src as usize + 1) * ch];
            for (i, v) in s.iter().enumerate() {
                row[i * 3 + j] = *v;
            }
        }
    }
    col
}

/// Rows 2k / 2k+1: periodic-Hann-windowed cos / sin of frequency bin k.
fn dft_basis() -> Vec<f32> {
    let mut b = vec![0.0; 2 * N_BINS * N_FFT];
    for k in 0..N_BINS {
        for n in 0..N_FFT {
            let w = 0.5 - 0.5 * (2.0 * PI * n as f64 / N_FFT as f64).cos();
            let a = 2.0 * PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
            b[2 * k * N_FFT + n] = (w * a.cos()) as f32;
            b[(2 * k + 1) * N_FFT + n] = (w * a.sin()) as f32;
        }
    }
    b
}

/// Whisper's fixed sinusoidal positions: [sin | cos].
fn sinusoids(len: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let inc = (10000f64).ln() / (half - 1) as f64;
    let mut p = vec![0.0; len * d];
    for t in 0..len {
        for i in 0..half {
            let a = t as f64 * (-inc * i as f64).exp();
            p[t * d + i] = a.sin() as f32;
            p[t * d + half + i] = a.cos() as f32;
        }
    }
    p
}

/// Windowed-sinc resampler (Hann window, 16 zero crossings), good enough for LID front ends.
/// Filter taps are precomputed per phase (rates reduced by their gcd, e.g. 48k -> 16k has one phase).
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let g = gcd(from as u64, to as u64);
    let (up, down) = ((to as u64 / g) as usize, (from as u64 / g) as usize);
    let cutoff = (to as f64 / from as f64).min(1.0); // relative to the input Nyquist
    let half = (16.0 / cutoff).ceil() as usize; // taps per side, in input samples
    let taps = 2 * half;
    // table[phase][j]: weight of input sample (c - half + 1 + j) for output position c + phase/up
    let phases = up.min(4096);
    let mut table = vec![0f32; phases * taps];
    for (ph, row) in table.chunks_exact_mut(taps).enumerate() {
        let frac = ph as f64 / phases as f64;
        let mut norm = 0.0;
        let mut w = vec![0f64; taps];
        for (j, w) in w.iter_mut().enumerate() {
            let dist = (j as f64 - half as f64 + 1.0) - frac;
            let x = dist * cutoff;
            let sinc = if x.abs() < 1e-9 { 1.0 } else { (PI * x).sin() / (PI * x) };
            let win = 0.5 + 0.5 * (PI * dist / half as f64).cos();
            *w = sinc * win.max(0.0);
            norm += *w;
        }
        for (r, w) in row.iter_mut().zip(&w) {
            *r = (w / norm) as f32;
        }
    }
    let out_len = (input.len() as u64 * up as u64 / down as u64) as usize;
    let mut out = Vec::with_capacity(out_len);
    for o in 0..out_len {
        let pos = o as u64 * down as u64; // in units of 1/up input samples
        let c = (pos / up as u64) as isize;
        let ph = ((pos % up as u64) as usize * phases) / up;
        let row = &table[ph * taps..(ph + 1) * taps];
        let start = c - half as isize + 1;
        let acc: f32 = if start >= 0 && (start as usize + taps) <= input.len() {
            kernels::dot(&input[start as usize..start as usize + taps], row)
        } else {
            // edges: drop taps outside the signal and renormalize
            let (mut acc, mut norm) = (0.0, 0.0);
            for (j, w) in row.iter().enumerate() {
                let i = start + j as isize;
                if i >= 0 && (i as usize) < input.len() {
                    acc += w * input[i as usize];
                    norm += w;
                }
            }
            if norm.abs() > 1e-9 { acc / norm } else { 0.0 }
        };
        out.push(acc);
    }
    out
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}
