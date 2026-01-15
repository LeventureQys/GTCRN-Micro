/**
 * @file esp32_example.c
 * @brief ESP32 GTCRN-Micro Usage Example
 *
 * This example shows how to use the GTCRN-Micro SDK on ESP32-S3.
 * It demonstrates:
 * - Platform initialization
 * - Engine creation with external memory (PSRAM)
 * - Model loading and inference
 * - LED status indication
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtcrn_micro/gtcrn_micro.h"
#include <stdio.h>

/* ESP32 specific includes */
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* Include your model data */
/* #include "gtcrn_micro_int8_data.cc" */
extern const unsigned char gtcrn_micro_full_integer_quant_tflite[];
extern const unsigned int gtcrn_micro_full_integer_quant_tflite_len;

/* GPIO for LED status indicator */
#define LED_PIN GPIO_NUM_38

/* Configuration */
#define ARENA_SIZE (300 * 1024)  /* 300KB for tensor arena */

static void led_init(void) {
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
}

static void led_set(int state) {
    gpio_set_level(LED_PIN, state);
}

static void blink_error(const char *msg) {
    GTCRN_LOGE("ESP32_EXAMPLE", "ERROR: %s", msg);
    while (1) {
        led_set(1);
        gtcrn_platform_delay_ms(100);
        led_set(0);
        gtcrn_platform_delay_ms(100);
    }
}

void app_main(void) {
    printf("\n");
    printf("========================================\n");
    printf("GTCRN-Micro SDK v%s - ESP32 Example\n", gtcrn_version());
    printf("========================================\n\n");

    led_init();

    /* Initialize platform */
    gtcrn_status_t status = gtcrn_platform_init();
    if (status != GTCRN_OK) {
        blink_error("Platform init failed");
        return;
    }

    /* Create engine with external memory (PSRAM) */
    gtcrn_engine_t *engine = NULL;
    gtcrn_engine_config_t config = {
        .arena_size = ARENA_SIZE,
        .mem_flags = GTCRN_MEM_EXTERNAL,  /* Use PSRAM */
        .use_external_arena = false,
        .external_arena = NULL,
    };

    status = gtcrn_engine_create(&config, &engine);
    if (status != GTCRN_OK) {
        blink_error("Engine create failed");
        return;
    }

    GTCRN_LOGI("ESP32_EXAMPLE", "Engine created, arena: %zu bytes", config.arena_size);

    /* Load model */
    status = gtcrn_engine_load_model(
        engine,
        gtcrn_micro_full_integer_quant_tflite,
        gtcrn_micro_full_integer_quant_tflite_len
    );
    if (status != GTCRN_OK) {
        blink_error("Model load failed");
        return;
    }

    /* Get input tensor info */
    gtcrn_tensor_info_t input_info;
    status = gtcrn_engine_get_input_info(engine, 0, &input_info);
    if (status == GTCRN_OK) {
        GTCRN_LOGI("ESP32_EXAMPLE", "Input: type=%d, dims=(%d,%d,%d,%d), bytes=%zu",
                   input_info.dtype,
                   input_info.dims[0], input_info.dims[1],
                   input_info.dims[2], input_info.dims[3],
                   input_info.bytes);
    }

    /* Prepare dummy input */
    if (input_info.dtype == GTCRN_DTYPE_INT8) {
        int8_t *data = (int8_t *)input_info.data;
        for (size_t i = 0; i < input_info.bytes; ++i) {
            data[i] = 0;
        }
    } else if (input_info.dtype == GTCRN_DTYPE_FLOAT32) {
        float *data = (float *)input_info.data;
        for (size_t i = 0; i < input_info.bytes / sizeof(float); ++i) {
            data[i] = 0.0f;
        }
    }

    /* Run inference */
    GTCRN_LOGI("ESP32_EXAMPLE", "Running inference...");
    status = gtcrn_engine_invoke(engine);
    if (status != GTCRN_OK) {
        blink_error("Inference failed");
        return;
    }

    /* Get statistics */
    gtcrn_stats_t stats;
    gtcrn_engine_get_stats(engine, &stats);
    GTCRN_LOGI("ESP32_EXAMPLE", "Inference time: %lld us", (long long)stats.inference_time_us);
    GTCRN_LOGI("ESP32_EXAMPLE", "Arena used: %zu / %zu bytes",
               stats.arena_used_bytes, stats.arena_total_bytes);

    /* Success - blink LED slowly */
    GTCRN_LOGI("ESP32_EXAMPLE", "Inference succeeded - blinking LED");

    int led_state = 0;
    while (1) {
        led_state = !led_state;
        led_set(led_state);
        printf("LED: %d\n", led_state);
        gtcrn_platform_delay_ms(500);
    }
}
