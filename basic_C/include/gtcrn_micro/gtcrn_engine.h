/**
 * @file gtcrn_engine.h
 * @brief GTCRN-Micro Inference Engine API
 *
 * This header provides the main API for loading and running GTCRN neural
 * network models using TensorFlow Lite Micro.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GTCRN_MICRO_ENGINE_H
#define GTCRN_MICRO_ENGINE_H

#include "gtcrn_config.h"
#include "gtcrn_platform.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup gtcrn_engine GTCRN Inference Engine
 * @brief Core API for loading and running GTCRN models.
 * @{
 */

/* ============================================================================
 * Types and Structures
 * ========================================================================== */

/**
 * @brief Opaque handle to a GTCRN engine instance.
 */
typedef struct gtcrn_engine gtcrn_engine_t;

/**
 * @brief Tensor data type enumeration.
 */
typedef enum {
    GTCRN_DTYPE_UNKNOWN = 0,
    GTCRN_DTYPE_FLOAT32 = 1,
    GTCRN_DTYPE_INT8 = 2,
    GTCRN_DTYPE_UINT8 = 3,
    GTCRN_DTYPE_INT16 = 4,
    GTCRN_DTYPE_INT32 = 5,
} gtcrn_dtype_t;

/**
 * @brief Tensor information structure.
 */
typedef struct {
    gtcrn_dtype_t dtype;   /**< Data type */
    int32_t dims[4];       /**< Dimensions (NHWC or similar) */
    int32_t num_dims;      /**< Number of dimensions */
    size_t bytes;          /**< Total size in bytes */
    void *data;            /**< Pointer to tensor data */

    /* Quantization parameters (for INT8 tensors) */
    float scale;           /**< Quantization scale */
    int32_t zero_point;    /**< Quantization zero point */
} gtcrn_tensor_info_t;

/**
 * @brief Engine configuration structure.
 */
typedef struct {
    size_t arena_size;           /**< Tensor arena size in bytes */
    uint32_t mem_flags;          /**< Memory allocation flags */
    bool use_external_arena;     /**< If true, use externally provided arena */
    uint8_t *external_arena;     /**< Pointer to external arena (if use_external_arena) */
} gtcrn_engine_config_t;

/**
 * @brief Inference statistics.
 */
typedef struct {
    int64_t inference_time_us;   /**< Last inference time in microseconds */
    size_t arena_used_bytes;     /**< Arena memory used */
    size_t arena_total_bytes;    /**< Total arena size */
    uint32_t inference_count;    /**< Total number of inferences run */
} gtcrn_stats_t;

/* ============================================================================
 * Initialization Macros
 * ========================================================================== */

/**
 * @brief Default engine configuration initializer.
 */
#define GTCRN_ENGINE_CONFIG_DEFAULT()               \
    {                                               \
        .arena_size = GTCRN_TENSOR_ARENA_SIZE,      \
        .mem_flags = GTCRN_MEM_DEFAULT,             \
        .use_external_arena = false,                \
        .external_arena = NULL,                     \
    }

/* ============================================================================
 * Engine Lifecycle
 * ========================================================================== */

/**
 * @brief Create a new GTCRN engine instance.
 *
 * @param config  Configuration options. Pass NULL for defaults.
 * @param engine  Output pointer to receive engine handle.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_create(const gtcrn_engine_config_t *config,
                                    gtcrn_engine_t **engine);

/**
 * @brief Destroy a GTCRN engine instance and free resources.
 *
 * @param engine Engine handle to destroy.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_destroy(gtcrn_engine_t *engine);

/* ============================================================================
 * Model Loading
 * ========================================================================== */

/**
 * @brief Load a TFLite model from a memory buffer.
 *
 * The model buffer must remain valid for the lifetime of the engine.
 *
 * @param engine     Engine handle.
 * @param model_data Pointer to TFLite model data (flatbuffer).
 * @param model_size Size of model data in bytes.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_load_model(gtcrn_engine_t *engine,
                                        const uint8_t *model_data,
                                        size_t model_size);

/**
 * @brief Check if a model is loaded and ready for inference.
 *
 * @param engine Engine handle.
 * @return true if model is loaded and ready, false otherwise.
 */
bool gtcrn_engine_is_ready(const gtcrn_engine_t *engine);

/* ============================================================================
 * Tensor Access
 * ========================================================================== */

/**
 * @brief Get input tensor count.
 *
 * @param engine Engine handle.
 * @return Number of input tensors.
 */
int gtcrn_engine_input_count(const gtcrn_engine_t *engine);

/**
 * @brief Get output tensor count.
 *
 * @param engine Engine handle.
 * @return Number of output tensors.
 */
int gtcrn_engine_output_count(const gtcrn_engine_t *engine);

/**
 * @brief Get input tensor information.
 *
 * @param engine Engine handle.
 * @param index  Input tensor index.
 * @param info   Output tensor info structure.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_get_input_info(const gtcrn_engine_t *engine,
                                            int index,
                                            gtcrn_tensor_info_t *info);

/**
 * @brief Get output tensor information.
 *
 * @param engine Engine handle.
 * @param index  Output tensor index.
 * @param info   Output tensor info structure.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_get_output_info(const gtcrn_engine_t *engine,
                                             int index,
                                             gtcrn_tensor_info_t *info);

/**
 * @brief Set input tensor data.
 *
 * Copies data into the input tensor buffer.
 *
 * @param engine Engine handle.
 * @param index  Input tensor index.
 * @param data   Source data pointer.
 * @param size   Size of data in bytes.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_set_input(gtcrn_engine_t *engine,
                                       int index,
                                       const void *data,
                                       size_t size);

/**
 * @brief Get output tensor data.
 *
 * Copies data from the output tensor buffer.
 *
 * @param engine Engine handle.
 * @param index  Output tensor index.
 * @param data   Destination data pointer.
 * @param size   Size of destination buffer in bytes.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_get_output(const gtcrn_engine_t *engine,
                                        int index,
                                        void *data,
                                        size_t size);

/* ============================================================================
 * Inference
 * ========================================================================== */

/**
 * @brief Run inference on the loaded model.
 *
 * Input data should be set before calling this function.
 * After completion, output data can be retrieved.
 *
 * @param engine Engine handle.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_invoke(gtcrn_engine_t *engine);

/* ============================================================================
 * Statistics and Diagnostics
 * ========================================================================== */

/**
 * @brief Get engine statistics.
 *
 * @param engine Engine handle.
 * @param stats  Output statistics structure.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_get_stats(const gtcrn_engine_t *engine,
                                       gtcrn_stats_t *stats);

/**
 * @brief Reset engine statistics.
 *
 * @param engine Engine handle.
 * @return GTCRN_OK on success, error code otherwise.
 */
gtcrn_status_t gtcrn_engine_reset_stats(gtcrn_engine_t *engine);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_MICRO_ENGINE_H */
