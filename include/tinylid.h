#ifndef TINYLID_H
#define TINYLID_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Spoken language identification (tinylid). One model instance is immutable after loading and
   can be shared between threads; tinylid starts no threads of its own. */
typedef struct TinylidModel TinylidModel;

/* Parses a model file (tinylid.bin). The bytes are copied; the buffer can be freed afterwards.
   Returns NULL if the data is not a valid model. */
TinylidModel *tinylid_load(const uint8_t *model, size_t len);
void tinylid_free(TinylidModel *m);

/* Number of output classes and their codes (ISO 639-1 where one exists, e.g. "uk", "en";
   "nospeech" in models trained with a silence class). The string lives as long as the model. */
uint32_t tinylid_num_languages(const TinylidModel *m);
const char *tinylid_language(const TinylidModel *m, uint32_t i);

/* Language probabilities for mono float audio in [-1, 1] at any sample rate (resampled to 16 kHz
   internally; only the first 30 s are used). probs_out must hold tinylid_num_languages() floats.
   exact = 1: pad to 30 s as in training (most accurate, fixed cost);
   exact = 0: only the real audio (faster for short clips, slightly less accurate).
   Returns 0 on success, -1 on invalid arguments. */
int32_t tinylid_predict(const TinylidModel *m, const float *audio, size_t n, uint32_t sample_rate,
                        int32_t exact, float *probs_out);

/* Whisper/Ear log-mel input of the first 30 s: out must hold 80 * 3000 floats (mel-major). */
int32_t tinylid_log_mel(const TinylidModel *m, const float *audio, size_t n, uint32_t sample_rate,
                        float *out);

#ifdef __wasm__
/* 16-byte aligned buffers in wasm memory, for a JS host to pass model bytes and audio. */
uint8_t *tinylid_alloc(size_t len);
void tinylid_dealloc(uint8_t *p, size_t len);
#endif

#ifdef __cplusplus
}
#endif

#endif
