/**
 * @file platform_esp32.c
 * @brief ESP32/ESP32-S3 Platform Implementation
 *
 * This file provides the platform abstraction layer implementation for
 * Espressif ESP32 series microcontrollers using ESP-IDF.
 *
 * To use this implementation:
 * 1. Include this file in your ESP-IDF project instead of platform_default.c
 * 2. Link against ESP-IDF components (freertos, esp_timer, etc.)
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtcrn_micro/gtcrn_platform.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ESP-IDF headers */
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* ============================================================================
 * Internal State
 * ========================================================================== */

static const char *TAG = "GTCRN_PAL_ESP32";
static int s_initialized = 0;

/* ============================================================================
 * Platform Initialization
 * ========================================================================== */

gtcrn_status_t gtcrn_platform_init(void) {
    if (s_initialized) {
        return GTCRN_ERROR_ALREADY;
    }

    ESP_LOGI(TAG, "GTCRN Platform initialized (ESP32)");
    ESP_LOGI(TAG, "Free internal heap: %d bytes", (int)esp_get_free_heap_size());
    ESP_LOGI(TAG, "Free SPIRAM: %d bytes",
             (int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    s_initialized = 1;
    return GTCRN_OK;
}

gtcrn_status_t gtcrn_platform_deinit(void) {
    if (!s_initialized) {
        return GTCRN_ERROR_NOT_INIT;
    }

    s_initialized = 0;
    ESP_LOGI(TAG, "GTCRN Platform deinitialized");
    return GTCRN_OK;
}

/* ============================================================================
 * Memory Management
 * ========================================================================== */

void *gtcrn_platform_malloc(size_t size, uint32_t flags) {
    if (size == 0) {
        return NULL;
    }

    uint32_t caps = MALLOC_CAP_8BIT;

    /* Map GTCRN flags to ESP-IDF heap capabilities */
    if (flags & GTCRN_MEM_EXTERNAL) {
        caps |= MALLOC_CAP_SPIRAM;
    } else {
        caps |= MALLOC_CAP_INTERNAL;
    }

    if (flags & GTCRN_MEM_DMA) {
        caps |= MALLOC_CAP_DMA;
    }

    void *ptr = heap_caps_malloc(size, caps);

    if (!ptr) {
        ESP_LOGE(TAG, "Failed to allocate %zu bytes with caps 0x%lx",
                 size, (unsigned long)caps);
    }

    return ptr;
}

void gtcrn_platform_free(void *ptr) {
    if (ptr) {
        heap_caps_free(ptr);
    }
}

size_t gtcrn_platform_get_free_memory(uint32_t flags) {
    if (flags & GTCRN_MEM_EXTERNAL) {
        return heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    } else {
        return esp_get_free_heap_size();
    }
}

/* ============================================================================
 * Timing
 * ========================================================================== */

int64_t gtcrn_platform_get_time_us(void) {
    return esp_timer_get_time();
}

uint32_t gtcrn_platform_get_time_ms(void) {
    return (uint32_t)(esp_timer_get_time() / 1000);
}

void gtcrn_platform_delay_ms(uint32_t ms) {
    vTaskDelay(ms / portTICK_PERIOD_MS);
}

/* ============================================================================
 * Logging
 * ========================================================================== */

void gtcrn_platform_log(gtcrn_log_level_t level, const char *tag,
                        const char *format, ...) {
    if (!format) return;

    esp_log_level_t esp_level;
    switch (level) {
    case GTCRN_LOG_ERROR:
        esp_level = ESP_LOG_ERROR;
        break;
    case GTCRN_LOG_WARN:
        esp_level = ESP_LOG_WARN;
        break;
    case GTCRN_LOG_INFO:
        esp_level = ESP_LOG_INFO;
        break;
    case GTCRN_LOG_DEBUG:
    default:
        esp_level = ESP_LOG_DEBUG;
        break;
    }

    va_list args;
    va_start(args, format);

    /* Use ESP_LOG_LEVEL_LOCAL for proper formatting */
    char buffer[GTCRN_MAX_LOG_LEN];
    vsnprintf(buffer, sizeof(buffer), format, args);

    switch (esp_level) {
    case ESP_LOG_ERROR:
        ESP_LOGE(tag ? tag : TAG, "%s", buffer);
        break;
    case ESP_LOG_WARN:
        ESP_LOGW(tag ? tag : TAG, "%s", buffer);
        break;
    case ESP_LOG_INFO:
        ESP_LOGI(tag ? tag : TAG, "%s", buffer);
        break;
    case ESP_LOG_DEBUG:
    default:
        ESP_LOGD(tag ? tag : TAG, "%s", buffer);
        break;
    }

    va_end(args);
}

/* ============================================================================
 * Critical Sections
 * ========================================================================== */

uint32_t gtcrn_platform_enter_critical(void) {
    taskENTER_CRITICAL_FROM_ISR();
    return 0;
}

void gtcrn_platform_exit_critical(uint32_t state) {
    (void)state;
    taskEXIT_CRITICAL_FROM_ISR(0);
}
