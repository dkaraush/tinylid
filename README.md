# tinylid

**Tiny** **L**anguage **Id**entification from audio speech for 125 languages, as 4.25MB quantized small model, fitted to be ran on device.

- **4.25 MB** model file: Whisper-tiny encoder with 4-bit weights, trained with quantization-aware training.
- Zero dependencies. `no_std` + `alloc` without the default `std` feature.
- SIMD: explicit NEON on ARM64, auto-vectorized SSE on x86-64 and SIMD128 on WebAssembly.
- Single-threaded, with no I/O. A loaded model is immutable, so the host can share it and owns all concurrency.
  The opt-in `threads` feature splits large matrix multiplies across cores on desktops.

### Building (Native)

```bash
cargo rustc --release --no-default-features --features c-api --lib --crate-type staticlib
# library will be built at target/release/libtinylid.a
# include headers are at include/tinylid.h
# shared library instead: --crate-type cdylib
```

### Modes

- `Exact` pads the clip to 30 s as in training. It is the most accurate and has a fixed cost.
- `Fast` encodes only the real audio. A 3 s clip costs about 10x less than in `Exact`. The current
  model was trained on padded input, so `Fast` loses about 1 point on 5 s clips and up to 16 points
  on 1–2 s clips.

### Speed (single thread)

| Device | Exact (any length) | Fast (11 s clip) |
|---|---|---|
| Samsung F15 / M15 (Cortex-A76) | 2.3 s | 0.72 s |
| WebAssembly, desktop Chrome | 0.56 s | 0.15 s |

Matrix multiplies run at about half the A76's fp32 peak. Most of the cost is the 30 s window,
not the kernels.
