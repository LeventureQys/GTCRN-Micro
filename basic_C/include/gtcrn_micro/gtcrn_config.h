/**
 * @file gtcrn_config.h
 * @brief GTCRN-Micro SDK Configuration
 *
 * This file contains compile-time configuration options for the GTCRN-Micro SDK.
 * Users can override these defaults by defining them before including this header.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GTCRN_MICRO_CONFIG_H
#define GTCRN_MICRO_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup gtcrn_config Configuration Options
 * @{
 */

/**
 * @brief Default tensor arena size in bytes.
 *
 * This is the amount of memory allocated for TensorFlow Lite Micro tensors.
 * Override this by defining GTCRN_TENSOR_ARENA_SIZE before including this header.
 */
#ifndef GTCRN_TENSOR_ARENA_SIZE
#define GTCRN_TENSOR_ARENA_SIZE (300 * 1024)
#endif

/**
 * @brief TFLite schema version expected by this SDK.
 */
#ifndef GTCRN_TFLITE_SCHEMA_VERSION
#define GTCRN_TFLITE_SCHEMA_VERSION 3
#endif

/**
 * @brief Number of TFLite operations registered.
 */
#ifndef GTCRN_NUM_OPS
#define GTCRN_NUM_OPS 24
#endif

/**
 * @brief Enable debug logging (1 = enabled, 0 = disabled).
 */
#ifndef GTCRN_DEBUG_LOG
#define GTCRN_DEBUG_LOG 1
#endif

/**
 * @brief Maximum log message length.
 */
#ifndef GTCRN_MAX_LOG_LEN
#define GTCRN_MAX_LOG_LEN 256
#endif

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_MICRO_CONFIG_H */
