//! C ABI for Android (JNI shim), iOS (static lib) and WebAssembly. Declared in include/tinylid.h;
//! the summary below uses void* for the model handle.
//!
//! ```c
//! void*  tinylid_load(const uint8_t* model, size_t len);          // NULL on error
//! void   tinylid_free(void* m);
//! uint32_t    tinylid_num_languages(const void* m);
//! const char* tinylid_language(const void* m, uint32_t i);       // NUL-terminated code, e.g. "uk"
//! int32_t tinylid_predict(const void* m, const float* audio, size_t n, uint32_t sample_rate,
//!                         int32_t exact, float* probs_out);     // probs_out[num_languages]; 0 = ok
//! int32_t tinylid_log_mel(const void* m, const float* audio, size_t n, uint32_t sample_rate,
//!                         float* out);                     // out[80 * 3000], Whisper/Ear input
//! uint8_t* tinylid_alloc(size_t len); void tinylid_dealloc(uint8_t* p, size_t len);  // wasm only
//! ```

#![allow(unsafe_code)]

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::{Mode, Model};

pub struct Handle {
    model: Model,
    names: Vec<Vec<u8>>, // NUL-terminated language codes
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinylid_load(data: *const u8, len: usize) -> *mut Handle {
    if data.is_null() {
        return core::ptr::null_mut();
    }
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    match Model::from_bytes(bytes) {
        Ok(model) => {
            let names = model.languages().iter().map(|l| l.bytes().chain([0]).collect()).collect();
            Box::into_raw(Box::new(Handle { model, names }))
        }
        Err(_) => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinylid_free(h: *mut Handle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinylid_num_languages(h: *const Handle) -> u32 {
    unsafe { h.as_ref() }.map_or(0, |h| h.names.len() as u32)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinylid_language(h: *const Handle, i: u32) -> *const u8 {
    unsafe { h.as_ref() }.and_then(|h| h.names.get(i as usize)).map_or(core::ptr::null(), |n| n.as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinylid_predict(
    h: *const Handle,
    audio: *const f32,
    n: usize,
    sample_rate: u32,
    exact: i32,
    probs_out: *mut f32,
) -> i32 {
    let Some(h) = (unsafe { h.as_ref() }) else { return -1 };
    if audio.is_null() || probs_out.is_null() || sample_rate == 0 {
        return -1;
    }
    let audio = unsafe { core::slice::from_raw_parts(audio, n) };
    let mode = if exact != 0 { Mode::Exact } else { Mode::Fast };
    let probs = h.model.predict_resampled(audio, sample_rate, mode);
    unsafe { core::slice::from_raw_parts_mut(probs_out, probs.len()) }.copy_from_slice(&probs);
    0
}

/// Whisper log-mel (n_mels x 3000, channel-major) of the first 30 s, for running Ear-style models.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinylid_log_mel(h: *const Handle, audio: *const f32, n: usize, sample_rate: u32, out: *mut f32) -> i32 {
    let Some(h) = (unsafe { h.as_ref() }) else { return -1 };
    if audio.is_null() || out.is_null() || sample_rate == 0 {
        return -1;
    }
    let audio = unsafe { core::slice::from_raw_parts(audio, n) };
    let audio = crate::resample(audio, sample_rate, crate::SAMPLE_RATE);
    let mel = h.model.log_mel_30s(&audio);
    unsafe { core::slice::from_raw_parts_mut(out, mel.len()) }.copy_from_slice(&mel);
    0
}

/// 16-byte aligned, so a JS host can place f32 audio in wasm memory. NULL if `len` is too large.
/// Native hosts use their own allocator, so these two exist only in wasm builds.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn tinylid_alloc(len: usize) -> *mut u8 {
    match layout(len) {
        Some(l) => unsafe { alloc::alloc::alloc(l) },
        None => core::ptr::null_mut(),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tinylid_dealloc(p: *mut u8, len: usize) {
    if let (false, Some(l)) = (p.is_null(), layout(len)) {
        unsafe { alloc::alloc::dealloc(p, l) };
    }
}

#[cfg(target_arch = "wasm32")]
fn layout(len: usize) -> Option<core::alloc::Layout> {
    core::alloc::Layout::from_size_align(len.max(1), 16).ok()
}
