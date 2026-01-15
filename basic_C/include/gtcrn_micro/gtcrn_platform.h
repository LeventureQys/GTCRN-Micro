/**
 * @file gtcrn_platform.h
 * @brief Platform Abstraction Layer (PAL) for GTCRN-Micro SDK
 *
 * This header defines the platform abstraction interface that must be
 * implemented for each target platform. It abstracts away platform-specific
 * functionality like memory allocation, timing, and logging.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GTCRN_MICRO_PLATFORM_H
#define GTCRN_MICRO_PLATFORM_H

#include "gtcrn_config.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup gtcrn_platform Platform Abstraction Layer
 * @brief Platform-specific functionality that must be implemented for each target.
 * @{
 */

/* ============================================================================
 * Error Codes
 * ========================================================================== */

/**
 * @brief GTCRN status codes.
 */
typedef enum {
    GTCRN_OK = 0,              /**< Operation successful */
    GTCRN_ERROR_GENERIC = -1,  /**< Generic error */
    GTCRN_ERROR_MEMORY = -2,   /**< Memory allocation failed */
    GTCRN_ERROR_INVALID = -3,  /**< Invalid parameter */
    GTCRN_ERROR_TIMEOUT = -4,  /**< Operation timeout */
    GTCRN_ERROR_NOT_INIT = -5, /**< Not initialized */
    GTCRN_ERROR_ALREADY = -6,  /**< Already initialized/running */
} gtcrn_status_t;

/**
 * @brief Log level enumeration.
 */
typedef enum {
    GTCRN_LOG_ERROR = 0, /**< Error messages */
    GTCRN_LOG_WARN = 1,  /**< Warning messages */
    GTCRN_LOG_INFO = 2,  /**< Informational messages */
    GTCRN_LOG_DEBUG = 3, /**< Debug messages */
} gtcrn_log_level_t;

/* ============================================================================
 * Memory Management
 * ========================================================================== */

/**
 * @brief Memory allocation flags.
 */
typedef enum {
    GTCRN_MEM_DEFAULT = 0,    /**< Default memory (internal RAM) */
    GTCRN_MEM_EXTERNAL = 1,   /**< External memory (PSRAM, etc.) */
    GTCRN_MEM_DMA = 2,        /**< DMA-capable memory */
    GTCRN_MEM_ALIGNED = 4,    /**< Aligned memory (platform-specific alignment) */
} gtcrn_mem_flags_t;

/**
 * @brief Allocate memory with specified flags.
 *
 * @param size  Size in bytes to allocate.
 * @param flags Memory allocation flags (gtcrn_mem_flags_t).
 * @return Pointer to allocated memory, or NULL on failure.
 */
void *gtcrn_platform_malloc(size_t size, uint32_t flags);

/**
 * @brief Free memory allocated by gtcrn_platform_malloc.
 *
 * @param ptr Pointer to memory to free.
 */
void gtcrn_platform_free(void *ptr);

/**
 * @brief Get available memory size.
 *
 * @param flags Memory type flags to query.
 * @return Available memory in bytes.
 */
size_t gtcrn_platform_get_free_memory(uint32_t flags);

/* ============================================================================
 * Timing
 * ========================================================================== */

/**
 * @brief Get current time in microseconds.
 *
 * @return Current time in microseconds since some reference point.
 */
int64_t gtcrn_platform_get_time_us(void);

/**
 * @brief Get current time in milliseconds.
 *
 * @return Current time in milliseconds since some reference point.
 */
uint32_t gtcrn_platform_get_time_ms(void);

/**
 * @brief Delay execution for specified milliseconds.
 *
 * @param ms Delay time in milliseconds.
 */
void gtcrn_platform_delay_ms(uint32_t ms);

/* ============================================================================
 * Logging
 * ========================================================================== */

/**
 * @brief Log a message with the specified level.
 *
 * @param level   Log level.
 * @param tag     Module/component tag.
 * @param format  Printf-style format string.
 * @param ...     Format arguments.
 */
void gtcrn_platform_log(gtcrn_log_level_t level, const char *tag,
                        const char *format, ...);

/**
 * @brief Convenience macros for logging.
 */
#if GTCRN_DEBUG_LOG
#define GTCRN_LOGE(tag, fmt, ...) \
    gtcrn_platform_log(GTCRN_LOG_ERROR, tag, fmt, ##__VA_ARGS__)
#define GTCRN_LOGW(tag, fmt, ...) \
    gtcrn_platform_log(GTCRN_LOG_WARN, tag, fmt, ##__VA_ARGS__)
#define GTCRN_LOGI(tag, fmt, ...) \
    gtcrn_platform_log(GTCRN_LOG_INFO, tag, fmt, ##__VA_ARGS__)
#define GTCRN_LOGD(tag, fmt, ...) \
    gtcrn_platform_log(GTCRN_LOG_DEBUG, tag, fmt, ##__VA_ARGS__)
#else
#define GTCRN_LOGE(tag, fmt, ...) ((void)0)
#define GTCRN_LOGW(tag, fmt, ...) ((void)0)
#define GTCRN_LOGI(tag, fmt, ...) ((void)0)
#define GTCRN_LOGD(tag, fmt, ...) ((void)0)
#endif

/* ============================================================================
 * Platform Initialization
 * ========================================================================== */

/**
 * @brief Initialize platform-specific resources.
 *
 * This function should be called once at startup before using any other
 * platform functions.
 *
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_platform_init(void);

/**
 * @brief Deinitialize platform-specific resources.
 *
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_platform_deinit(void);

/* ============================================================================
 * Critical Sections (Optional)
 * ========================================================================== */

/**
 * @brief Enter critical section (disable interrupts).
 *
 * @return Opaque state value to pass to gtcrn_platform_exit_critical.
 */
uint32_t gtcrn_platform_enter_critical(void);

/**
 * @brief Exit critical section (restore interrupts).
 *
 * @param state State value returned by gtcrn_platform_enter_critical.
 */
void gtcrn_platform_exit_critical(uint32_t state);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_MICRO_PLATFORM_H */
