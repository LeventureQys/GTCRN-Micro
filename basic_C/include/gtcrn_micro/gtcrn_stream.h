/**
 * @file gtcrn_stream.h
 * @brief GTCRN-Micro Streaming Audio Processor
 */

#ifndef GTCRN_STREAM_H
#define GTCRN_STREAM_H

#include "gtcrn_engine.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model parameters from gtcrn_micro_stream.py */
#define GTCRN_NFFT          512
#define GTCRN_HOP_LEN       256
#define GTCRN_WIN_LEN       512
#define GTCRN_FREQ_BINS     257   /* NFFT/2 + 1 */
#define GTCRN_SAMPLE_RATE   16000

/* Cache dimensions from streaming model */
#define GTCRN_CONV_CACHE_DIM0   2
#define GTCRN_CONV_CACHE_DIM1   1
#define GTCRN_CONV_CACHE_DIM2   16
#define GTCRN_CONV_CACHE_DIM3   6
#define GTCRN_CONV_CACHE_DIM4   33

#define GTCRN_TRA_CACHE_DIM0    2
#define GTCRN_TRA_CACHE_DIM1    3
#define GTCRN_TRA_CACHE_DIM2    1
#define GTCRN_TRA_CACHE_DIM3    8
#define GTCRN_TRA_CACHE_DIM4    2

/* TCN cache sizes: 8 caches with varying temporal dimensions */
#define GTCRN_TCN_CACHE_COUNT   8

/**
 * @brief Opaque handle to streaming processor
 */
typedef struct gtcrn_stream gtcrn_stream_t;

/**
 * @brief Create streaming processor
 * @param engine Initialized GTCRN engine with loaded model
 * @param stream Output stream handle
 * @return GTCRN_OK on success
 */
gtcrn_status_t gtcrn_stream_create(gtcrn_engine_t *engine, gtcrn_stream_t **stream);

/**
 * @brief Destroy streaming processor
 */
gtcrn_status_t gtcrn_stream_destroy(gtcrn_stream_t *stream);

/**
 * @brief Reset all internal caches to zero
 */
gtcrn_status_t gtcrn_stream_reset(gtcrn_stream_t *stream);

/**
 * @brief Process one frame of complex spectrum
 * @param stream Stream handle
 * @param spec_in Input spectrum [FREQ_BINS, 1, 2] (real, imag)
 * @param spec_out Output enhanced spectrum [FREQ_BINS, 1, 2]
 * @return GTCRN_OK on success
 */
gtcrn_status_t gtcrn_stream_process_frame(gtcrn_stream_t *stream,
                                           const float *spec_in,
                                           float *spec_out);

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_STREAM_H */
