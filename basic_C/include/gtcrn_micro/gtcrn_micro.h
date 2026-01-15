/**
 * @file gtcrn_micro.h
 * @brief GTCRN-Micro SDK - Main Include Header
 *
 * This is the main header file for the GTCRN-Micro SDK.
 * Include this single file to access all SDK functionality.
 *
 * @example Basic Usage:
 * @code
 * #include "gtcrn_micro/gtcrn_micro.h"
 *
 * // Initialize platform
 * gtcrn_platform_init();
 *
 * // Create engine
 * gtcrn_engine_t *engine;
 * gtcrn_engine_config_t config = GTCRN_ENGINE_CONFIG_DEFAULT();
 * gtcrn_engine_create(&config, &engine);
 *
 * // Load model
 * gtcrn_engine_load_model(engine, model_data, model_size);
 *
 * // Set input and run inference
 * gtcrn_engine_set_input(engine, 0, input_data, input_size);
 * gtcrn_engine_invoke(engine);
 *
 * // Get output
 * gtcrn_engine_get_output(engine, 0, output_data, output_size);
 *
 * // Cleanup
 * gtcrn_engine_destroy(engine);
 * gtcrn_platform_deinit();
 * @endcode
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GTCRN_MICRO_H
#define GTCRN_MICRO_H

/* SDK Version */
#define GTCRN_MICRO_VERSION_MAJOR 1
#define GTCRN_MICRO_VERSION_MINOR 0
#define GTCRN_MICRO_VERSION_PATCH 0
#define GTCRN_MICRO_VERSION_STRING "1.0.0"

/* Core Headers */
#include "gtcrn_config.h"
#include "gtcrn_platform.h"
#include "gtcrn_engine.h"
#include "gtcrn_stream.h"
#include "gtcrn_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get SDK version string.
 *
 * @return Version string in format "major.minor.patch".
 */
const char *gtcrn_version(void);

/**
 * @brief Get SDK version as numeric components.
 *
 * @param major Output major version (can be NULL).
 * @param minor Output minor version (can be NULL).
 * @param patch Output patch version (can be NULL).
 */
void gtcrn_version_info(int *major, int *minor, int *patch);

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_MICRO_H */
