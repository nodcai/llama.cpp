#pragma once

#include "ggml.h"
#include "clip-model.h"

#include <cstdint>
#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

struct mtmd_audio_mel {
    int64_t n_len;
    int64_t n_len_org;
    int64_t n_mel;

    std::vector<float> data;
};

struct mtmd_audio_mel_filters {
    int64_t n_mel;
    int64_t n_fft;

    std::vector<float> data;
};

// cache for audio processing, each processor instance owns its own cache
struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    std::vector<float> hann_window;

    mtmd_audio_mel_filters filters;

    void fill_sin_cos_table(uint32_t n);

    void fill_hann_window(uint32_t length, bool periodic);

    // Build mel filterbank matrix [n_mel × n_fft_bins] at runtime.
    // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
    void fill_mel_filterbank_matrix(int64_t n_mel,
                                    int64_t n_fft,
                                    int   sample_rate,               // e.g. 16000
                                    float fmin             = 0.0f,   // e.g. 0.0
                                    float fmax             = -1.0f,  // e.g. sr/2; pass -1 for auto
                                    bool  slaney_area_norm = true,
                                    float scale            = 1.0f,
                                    bool  use_htk          = false
    );
};

struct mtmd_audio_preprocessor {
    const clip_hparams & hparams;

    mtmd_audio_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}
    mtmd_audio_preprocessor(const clip_hparams & hparams): hparams(hparams) {}

    virtual ~mtmd_audio_preprocessor() = default;
    virtual void initialize() = 0; // NOT thread-safe
    virtual bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const = 0;
};

struct mtmd_audio_preprocessor_whisper : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_whisper(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_conformer : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_conformer(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    mtmd_audio_preprocessor_conformer(const clip_hparams & hparams) : mtmd_audio_preprocessor(hparams) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_granite_speech : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_granite_speech(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_gemma4a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_gemma4a(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_gemma4ua : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_gemma4ua(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;
};

// Stateful streaming mel preprocessor for gemma4a (mtmd_audio_stream backend).
// Holds the cumulative PCM with the semicausal left-pad applied once at start,
// and emits new mel frames in O(new_pcm_samples) per push by feeding only the
// unseen slice to log_mel_spectrogram. Frame values are bit-identical to the
// one-shot gemma4a preprocessor: the Hann window + FFT + mel filterbank for a
// frame at offset O depend only on samples [O, O+frame_size), so slicing the
// buffer at a hop boundary doesn't change the math. The trailing "tentative"
// frame (whose mel window would extend past the cumulative PCM) is held back
// until finalize(), which pads with zeros to match the one-shot path's
// right-pad semantics exactly. Assumes audio < 30 s (one mel chunk).
struct mtmd_audio_stream_preproc_gemma4a {
    explicit mtmd_audio_stream_preproc_gemma4a(const clip_ctx * ctx);

    void   push(const float * samples, size_t n_samples);
    // Build the encoder-ready mel buffer (mel-major [n_mel][n_emitted()]).
    // When `with_right_pad` is true, also emits the trailing one frame that
    // matches the one-shot preprocessor's pt_frames count using zero right-pad
    // — call this only at finalize().
    void   get_mel(mtmd_audio_mel & out, bool with_right_pad);
    int    n_emitted_frames() const { return n_emitted; }
    size_t n_pcm_samples() const;   // cumulative real PCM seen (excluding pad)
    void   reset();

  private:
    const clip_hparams & hparams;
    mtmd_audio_cache     cache;

    bool               initialized = false;
    std::vector<float> pcm;       // [pad_left zeros] + cumulative real PCM
    std::vector<float> mel_time_major; // [n_emitted * n_mel] time-major
    int                n_emitted = 0;
};

struct mtmd_audio_preprocessor_qwen3a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_qwen3a(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_dots3note : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_dots3note(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_mimo_audio : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_mimo_audio(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_qwen3tts_spk : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_qwen3tts_spk(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;
};

// mimi convolves the waveform directly, so this only pads it to a whole number of frames
struct mtmd_audio_preprocessor_pockettts : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_pockettts(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override {}
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;
};

struct mtmd_audio_preprocessor_parakeet : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_parakeet(clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) { }
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) const override;

  private:
    mtmd_audio_cache cache;

    static void worker_thread(int ith, const float * window_func, int window_size,
                              const std::vector<float> & samples, int n_samples,
                              int frame_size, int frame_step, int n_threads,
                              int n_fft_bins,
                              const mtmd_audio_cache & cache, mtmd_audio_mel & mel);
};

//
// streaming ISTFT - converts spectrogram frames back to audio one frame at a time
//
struct mtmd_audio_streaming_istft {
    mtmd_audio_streaming_istft(int n_fft, int hop_length);

    // reset streaming state
    void reset();

    // process a single STFT frame (streaming)
    // frame_spectrum: [n_fft_bins x 2] interleaved real/imag
    // returns: up to hop_length samples
    std::vector<float> process_frame(const float * frame_spectrum);

    // flush remaining samples at end of stream
    std::vector<float> flush();

  private:
    int n_fft;
    int hop_length;
    int n_fft_bins;

    // Own cache for output processing
    mtmd_audio_cache cache;

    // Streaming state
    std::vector<float> overlap_buffer;
    std::vector<float> window_sum_buffer;
    int                padding_to_remove;

    // Working buffers for IFFT
    std::vector<float> ifft_in;
    std::vector<float> ifft_out;
};
