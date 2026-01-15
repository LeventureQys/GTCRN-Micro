/**
 * @file platform_default.c
 * @brief Default Platform Implementation (Standard C/C++)
 *
 * This file provides a default platform implementation using only standard
 * C/C++ library functions. It works on most POSIX-compatible systems and
 * Windows.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtcrn_micro/gtcrn_platform.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Platform-specific includes for timing */
#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

/* ============================================================================
 * Internal State
 * ========================================================================== */

static int s_initialized = 0;

#if defined(_WIN32) || defined(_WIN64)
static LARGE_INTEGER s_frequency;
static LARGE_INTEGER s_start_time;
#else
static struct timespec s_start_time;
#endif

/* ============================================================================
 * Platform Initialization
 * ========================================================================== */

gtcrn_status_t gtcrn_platform_init(void) {
    if (s_initialized) {
        return GTCRN_ERROR_ALREADY;
    }

#if defined(_WIN32) || defined(_WIN64)
    QueryPerformanceFrequency(&s_frequency);
    QueryPerformanceCounter(&s_start_time);
#else
    clock_gettime(CLOCK_MONOTONIC, &s_start_time);
#endif

    s_initialized = 1;
    return GTCRN_OK;
}

gtcrn_status_t gtcrn_platform_deinit(void) {
    if (!s_initialized) {
        return GTCRN_ERROR_NOT_INIT;
    }
    s_initialized = 0;
    return GTCRN_OK;
}

/* ============================================================================
 * Memory Management
 * ========================================================================== */

void *gtcrn_platform_malloc(size_t size, uint32_t flags) {
    (void)flags; /* Standard malloc doesn't support special flags */

    if (size == 0) {
        return NULL;
    }

    /* For aligned memory, use platform-specific aligned allocation */
    if (flags & GTCRN_MEM_ALIGNED) {
#if defined(_WIN32) || defined(_WIN64)
        return _aligned_malloc(size, 16);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
        return aligned_alloc(16, (size + 15) & ~15);
#else
        /* Fallback: allocate extra space and align manually */
        void *ptr = malloc(size + 16 + sizeof(void *));
        if (!ptr) return NULL;
        void **aligned = (void **)(((uintptr_t)ptr + 16 + sizeof(void *)) & ~15);
        aligned[-1] = ptr;
        return aligned;
#endif
    }

    return malloc(size);
}

void gtcrn_platform_free(void *ptr) {
    if (ptr) {
#if defined(_WIN32) || defined(_WIN64)
        /* Note: Can't distinguish aligned vs regular malloc on Windows without tracking */
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

size_t gtcrn_platform_get_free_memory(uint32_t flags) {
    (void)flags;
    /* Standard C has no portable way to query available memory */
    /* Return a large value to indicate "unknown/plenty" */
    return (size_t)-1;
}

/* ============================================================================
 * Timing
 * ========================================================================== */

int64_t gtcrn_platform_get_time_us(void) {
#if defined(_WIN32) || defined(_WIN64)
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (int64_t)((now.QuadPart - s_start_time.QuadPart) * 1000000 /
                     s_frequency.QuadPart);
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t us = (now.tv_sec - s_start_time.tv_sec) * 1000000;
    us += (now.tv_nsec - s_start_time.tv_nsec) / 1000;
    return us;
#endif
}

uint32_t gtcrn_platform_get_time_ms(void) {
    return (uint32_t)(gtcrn_platform_get_time_us() / 1000);
}

void gtcrn_platform_delay_ms(uint32_t ms) {
#if defined(_WIN32) || defined(_WIN64)
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}

/* ============================================================================
 * Logging
 * ========================================================================== */

static const char *s_log_level_names[] = {"ERROR", "WARN", "INFO", "DEBUG"};

void gtcrn_platform_log(gtcrn_log_level_t level, const char *tag,
                        const char *format, ...) {
    if (!format) return;

    const char *level_str =
        (level >= 0 && level <= GTCRN_LOG_DEBUG) ? s_log_level_names[level] : "???";

    fprintf(stderr, "[%s] %s: ", level_str, tag ? tag : "GTCRN");

    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
}

/* ============================================================================
 * Critical Sections
 * ========================================================================== */

uint32_t gtcrn_platform_enter_critical(void) {
    /* Single-threaded default implementation - no-op */
    return 0;
}

void gtcrn_platform_exit_critical(uint32_t state) {
    (void)state;
    /* Single-threaded default implementation - no-op */
}
