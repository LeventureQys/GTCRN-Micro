/**
 * @file simple_inference.cpp
 * @brief GTCRN-Micro Streaming Inference Example
 */

#include "gtcrn_micro/gtcrn_micro.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Simple Hann window */
static void hann_window(float *win, int len) {
    for (int i = 0; i < len; i++) {
        win[i] = sqrtf(0.5f * (1.0f - cosf(2.0f * M_PI * i / len)));
    }
}

int main(int argc, char *argv[]) {
    printf("GTCRN-Micro SDK v%s\n", gtcrn_version());
    printf("========================================\n\n");

    /* Initialize platform */
    gtcrn_status_t status = gtcrn_platform_init();
    if (status != GTCRN_OK) {
        fprintf(stderr, "Failed to initialize platform: %d\n", status);
        return 1;
    }

    /* Create engine */
    gtcrn_engine_t *engine = NULL;
    gtcrn_engine_config_t config = GTCRN_ENGINE_CONFIG_DEFAULT();

    status = gtcrn_engine_create(&config, &engine);
    if (status != GTCRN_OK) {
        fprintf(stderr, "Failed to create engine: %d\n", status);
        gtcrn_platform_deinit();
        return 1;
    }

    printf("Engine created (arena: %zu bytes)\n", config.arena_size);

    /* Load model */
    status = gtcrn_engine_load_model(engine, gtcrn_model_data, gtcrn_model_data_len);
    if (status != GTCRN_OK) {
        fprintf(stderr, "Failed to load model: %d\n", status);
        gtcrn_engine_destroy(engine);
        gtcrn_platform_deinit();
        return 1;
    }

    printf("Model loaded (%u bytes)\n", gtcrn_model_data_len);

    /* Print tensor info */
    int n_inputs = gtcrn_engine_input_count(engine);
    int n_outputs = gtcrn_engine_output_count(engine);
    printf("Inputs: %d, Outputs: %d\n\n", n_inputs, n_outputs);

    for (int i = 0; i < n_inputs; i++) {
        gtcrn_tensor_info_t info;
        if (gtcrn_engine_get_input_info(engine, i, &info) == GTCRN_OK) {
            printf("Input %d: [%d,%d,%d,%d] %zu bytes\n",
                   i, info.dims[0], info.dims[1], info.dims[2], info.dims[3], info.bytes);
        }
    }

    for (int i = 0; i < n_outputs; i++) {
        gtcrn_tensor_info_t info;
        if (gtcrn_engine_get_output_info(engine, i, &info) == GTCRN_OK) {
            printf("Output %d: [%d,%d,%d,%d] %zu bytes\n",
                   i, info.dims[0], info.dims[1], info.dims[2], info.dims[3], info.bytes);
        }
    }

    /* Create streaming processor */
    gtcrn_stream_t *stream = NULL;
    status = gtcrn_stream_create(engine, &stream);
    if (status != GTCRN_OK) {
        fprintf(stderr, "Failed to create stream processor: %d\n", status);
        gtcrn_engine_destroy(engine);
        gtcrn_platform_deinit();
        return 1;
    }

    printf("\nStream processor created\n");

    /* Test with dummy data */
    float spec_in[GTCRN_FREQ_BINS * 2];   /* [257, 1, 2] flattened */
    float spec_out[GTCRN_FREQ_BINS * 2];

    memset(spec_in, 0, sizeof(spec_in));

    printf("Running test inference...\n");
    status = gtcrn_stream_process_frame(stream, spec_in, spec_out);
    if (status == GTCRN_OK) {
        printf("Inference OK!\n");

        gtcrn_stats_t stats;
        if (gtcrn_engine_get_stats(engine, &stats) == GTCRN_OK) {
            printf("Time: %lld us, Arena: %zu/%zu bytes\n",
                   (long long)stats.inference_time_us,
                   stats.arena_used_bytes, stats.arena_total_bytes);
        }
    } else {
        fprintf(stderr, "Inference failed: %d\n", status);
    }

    /* Cleanup */
    gtcrn_stream_destroy(stream);
    gtcrn_engine_destroy(engine);
    gtcrn_platform_deinit();

    printf("\nDone!\n");
    return 0;
}
