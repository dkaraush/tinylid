# tinylid

**Tiny** **L**anguage **Id**entification from audio speech for 15 languages, as 4.22MB quantized small model, fitted to be ran on device.

- **4.22 MB** model file (v12): Whisper-tiny encoder (4 layers) with 4-bit weights, trained with quantization-aware training.
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
