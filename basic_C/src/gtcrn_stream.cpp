/**
 * @file gtcrn_stream.cpp
 * @brief GTCRN-Micro Streaming Processor Implementation
 */

#include "gtcrn_micro/gtcrn_stream.h"
#include "gtcrn_micro/gtcrn_platform.h"
#include <cstring>

static const char *TAG = "GTCRN_STREAM";

/* TCN cache temporal dimensions: [2, 4, 8, 16, 2, 4, 8, 16] */
static const int tcn_cache_t_dims[GTCRN_TCN_CACHE_COUNT] = {2, 4, 8, 16, 2, 4, 8, 16};

struct gtcrn_stream {
    gtcrn_engine_t *engine;

    /* Caches - allocated as flat arrays */
    float *conv_cache;   /* [2, 1, 16, 6, 33] */
    float *tra_cache;    /* [2, 3, 1, 8, 2] */
    float *tcn_caches[GTCRN_TCN_CACHE_COUNT];  /* [1, 16, T, 33] each */
};

/* Cache sizes in floats */
static size_t conv_cache_size(void) {
    return GTCRN_CONV_CACHE_DIM0 * GTCRN_CONV_CACHE_DIM1 *
           GTCRN_CONV_CACHE_DIM2 * GTCRN_CONV_CACHE_DIM3 * GTCRN_CONV_CACHE_DIM4;
}

static size_t tra_cache_size(void) {
    return GTCRN_TRA_CACHE_DIM0 * GTCRN_TRA_CACHE_DIM1 *
           GTCRN_TRA_CACHE_DIM2 * GTCRN_TRA_CACHE_DIM3 * GTCRN_TRA_CACHE_DIM4;
}

static size_t tcn_cache_size(int idx) {
    return 1 * 16 * tcn_cache_t_dims[idx] * 33;
}

extern "C" gtcrn_status_t gtcrn_stream_create(gtcrn_engine_t *engine,
                                               gtcrn_stream_t **stream) {
    if (!engine || !stream) return GTCRN_ERROR_INVALID;

    gtcrn_stream_t *s = static_cast<gtcrn_stream_t*>(
        gtcrn_platform_malloc(sizeof(gtcrn_stream_t), GTCRN_MEM_DEFAULT));
    if (!s) return GTCRN_ERROR_MEMORY;

    std::memset(s, 0, sizeof(gtcrn_stream_t));
    s->engine = engine;

    /* Allocate caches */
    s->conv_cache = static_cast<float*>(
        gtcrn_platform_malloc(conv_cache_size() * sizeof(float), GTCRN_MEM_DEFAULT));
    s->tra_cache = static_cast<float*>(
        gtcrn_platform_malloc(tra_cache_size() * sizeof(float), GTCRN_MEM_DEFAULT));

    if (!s->conv_cache || !s->tra_cache) {
        gtcrn_stream_destroy(s);
        return GTCRN_ERROR_MEMORY;
    }

    for (int i = 0; i < GTCRN_TCN_CACHE_COUNT; i++) {
        s->tcn_caches[i] = static_cast<float*>(
            gtcrn_platform_malloc(tcn_cache_size(i) * sizeof(float), GTCRN_MEM_DEFAULT));
        if (!s->tcn_caches[i]) {
            gtcrn_stream_destroy(s);
            return GTCRN_ERROR_MEMORY;
        }
    }

    gtcrn_stream_reset(s);
    *stream = s;

    GTCRN_LOGI(TAG, "Stream processor created");
    return GTCRN_OK;
}

extern "C" gtcrn_status_t gtcrn_stream_destroy(gtcrn_stream_t *stream) {
    if (!stream) return GTCRN_ERROR_INVALID;

    if (stream->conv_cache) gtcrn_platform_free(stream->conv_cache);
    if (stream->tra_cache) gtcrn_platform_free(stream->tra_cache);

    for (int i = 0; i < GTCRN_TCN_CACHE_COUNT; i++) {
        if (stream->tcn_caches[i]) gtcrn_platform_free(stream->tcn_caches[i]);
    }

    gtcrn_platform_free(stream);
    return GTCRN_OK;
}

extern "C" gtcrn_status_t gtcrn_stream_reset(gtcrn_stream_t *stream) {
    if (!stream) return GTCRN_ERROR_INVALID;

    std::memset(stream->conv_cache, 0, conv_cache_size() * sizeof(float));
    std::memset(stream->tra_cache, 0, tra_cache_size() * sizeof(float));

    for (int i = 0; i < GTCRN_TCN_CACHE_COUNT; i++) {
        std::memset(stream->tcn_caches[i], 0, tcn_cache_size(i) * sizeof(float));
    }

    return GTCRN_OK;
}

extern "C" gtcrn_status_t gtcrn_stream_process_frame(gtcrn_stream_t *stream,
                                                      const float *spec_in,
                                                      float *spec_out) {
    if (!stream || !spec_in || !spec_out) return GTCRN_ERROR_INVALID;

    /*
     * Model inputs (from ONNX):
     * - audio: [1, 257, 1, 2]
     * - conv_cache: [2, 1, 16, 6, 33]
     * - tra_cache: [2, 3, 1, 8, 2]
     * - tcn_cache_0..7: [1, 16, T, 33]
     */

    /* Set inputs */
    size_t spec_size = GTCRN_FREQ_BINS * 1 * 2 * sizeof(float);
    gtcrn_engine_set_input(stream->engine, 0, spec_in, spec_size);
    gtcrn_engine_set_input(stream->engine, 1, stream->conv_cache,
                           conv_cache_size() * sizeof(float));
    gtcrn_engine_set_input(stream->engine, 2, stream->tra_cache,
                           tra_cache_size() * sizeof(float));

    for (int i = 0; i < GTCRN_TCN_CACHE_COUNT; i++) {
        gtcrn_engine_set_input(stream->engine, 3 + i, stream->tcn_caches[i],
                               tcn_cache_size(i) * sizeof(float));
    }

    /* Run inference */
    gtcrn_status_t status = gtcrn_engine_invoke(stream->engine);
    if (status != GTCRN_OK) return status;

    /* Get outputs */
    gtcrn_engine_get_output(stream->engine, 0, spec_out, spec_size);
    gtcrn_engine_get_output(stream->engine, 1, stream->conv_cache,
                            conv_cache_size() * sizeof(float));
    gtcrn_engine_get_output(stream->engine, 2, stream->tra_cache,
                            tra_cache_size() * sizeof(float));

    for (int i = 0; i < GTCRN_TCN_CACHE_COUNT; i++) {
        gtcrn_engine_get_output(stream->engine, 3 + i, stream->tcn_caches[i],
                                tcn_cache_size(i) * sizeof(float));
    }

    return GTCRN_OK;
}
