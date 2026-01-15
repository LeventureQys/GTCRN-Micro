/**
 * @file gtcrn_engine.cpp
 * @brief GTCRN-Micro Inference Engine Implementation
 *
 * This file implements the GTCRN inference engine using TensorFlow Lite Micro.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtcrn_micro/gtcrn_engine.h"
#include "gtcrn_micro/gtcrn_platform.h"

#include <cstring>

/* TensorFlow Lite Micro headers */
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ============================================================================
 * Internal Structures
 * ========================================================================== */

static const char *TAG = "GTCRN_ENGINE";

struct gtcrn_engine {
    /* Configuration */
    gtcrn_engine_config_t config;

    /* Memory */
    uint8_t *arena;
    bool arena_owned;

    /* TFLite Micro */
    const tflite::Model *model;
    tflite::MicroMutableOpResolver<GTCRN_NUM_OPS> *resolver;
    tflite::MicroInterpreter *interpreter;

    /* State */
    bool is_ready;

    /* Statistics */
    gtcrn_stats_t stats;
};

/* ============================================================================
 * Helper Functions
 * ========================================================================== */

static gtcrn_dtype_t tflite_type_to_gtcrn(TfLiteType type) {
    switch (type) {
    case kTfLiteFloat32:
        return GTCRN_DTYPE_FLOAT32;
    case kTfLiteInt8:
        return GTCRN_DTYPE_INT8;
    case kTfLiteUInt8:
        return GTCRN_DTYPE_UINT8;
    case kTfLiteInt16:
        return GTCRN_DTYPE_INT16;
    case kTfLiteInt32:
        return GTCRN_DTYPE_INT32;
    default:
        return GTCRN_DTYPE_UNKNOWN;
    }
}

static void fill_tensor_info(const TfLiteTensor *tensor, gtcrn_tensor_info_t *info) {
    if (!tensor || !info) return;

    info->dtype = tflite_type_to_gtcrn(tensor->type);
    info->num_dims = tensor->dims->size;
    info->bytes = tensor->bytes;
    info->data = tensor->data.raw;

    /* Fill dimensions (up to 4) */
    for (int i = 0; i < 4; ++i) {
        if (i < info->num_dims) {
            info->dims[i] = tensor->dims->data[i];
        } else {
            info->dims[i] = 0;
        }
    }

    /* Quantization parameters */
    if (tensor->quantization.type == kTfLiteAffineQuantization) {
        const TfLiteAffineQuantization *quant =
            static_cast<const TfLiteAffineQuantization *>(tensor->quantization.params);
        if (quant && quant->scale && quant->scale->size > 0) {
            info->scale = quant->scale->data[0];
        } else {
            info->scale = 1.0f;
        }
        if (quant && quant->zero_point && quant->zero_point->size > 0) {
            info->zero_point = quant->zero_point->data[0];
        } else {
            info->zero_point = 0;
        }
    } else {
        info->scale = 1.0f;
        info->zero_point = 0;
    }
}

static gtcrn_status_t register_ops(tflite::MicroMutableOpResolver<GTCRN_NUM_OPS> *resolver) {
    if (!resolver) return GTCRN_ERROR_INVALID;

    /* Register all required operations for GTCRN model */
    resolver->AddGather();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddTranspose();
    resolver->AddReshape();
    resolver->AddDequantize();
    resolver->AddSqrt();
    resolver->AddQuantize();
    resolver->AddConcatenation();
    resolver->AddStridedSlice();
    resolver->AddFullyConnected();
    resolver->AddConv2D();
    resolver->AddAbs();
    resolver->AddSub();
    resolver->AddRelu();
    resolver->AddPad();
    resolver->AddDepthwiseConv2D();
    resolver->AddTransposeConv();
    resolver->AddTanh();

    return GTCRN_OK;
}

/* ============================================================================
 * Engine Lifecycle
 * ========================================================================== */

extern "C" gtcrn_status_t gtcrn_engine_create(const gtcrn_engine_config_t *config,
                                               gtcrn_engine_t **engine) {
    if (!engine) {
        return GTCRN_ERROR_INVALID;
    }

    /* Allocate engine structure */
    gtcrn_engine_t *eng = static_cast<gtcrn_engine_t *>(
        gtcrn_platform_malloc(sizeof(gtcrn_engine_t), GTCRN_MEM_DEFAULT));

    if (!eng) {
        GTCRN_LOGE(TAG, "Failed to allocate engine structure");
        return GTCRN_ERROR_MEMORY;
    }

    std::memset(eng, 0, sizeof(gtcrn_engine_t));

    /* Apply configuration */
    if (config) {
        eng->config = *config;
    } else {
        eng->config.arena_size = GTCRN_TENSOR_ARENA_SIZE;
        eng->config.mem_flags = GTCRN_MEM_DEFAULT;
        eng->config.use_external_arena = false;
        eng->config.external_arena = nullptr;
    }

    /* Allocate tensor arena */
    if (eng->config.use_external_arena && eng->config.external_arena) {
        eng->arena = eng->config.external_arena;
        eng->arena_owned = false;
    } else {
        eng->arena = static_cast<uint8_t *>(
            gtcrn_platform_malloc(eng->config.arena_size, eng->config.mem_flags));
        if (!eng->arena) {
            GTCRN_LOGE(TAG, "Failed to allocate tensor arena (%zu bytes)",
                       eng->config.arena_size);
            gtcrn_platform_free(eng);
            return GTCRN_ERROR_MEMORY;
        }
        eng->arena_owned = true;
    }

    eng->stats.arena_total_bytes = eng->config.arena_size;

    GTCRN_LOGI(TAG, "Engine created with arena size %zu bytes", eng->config.arena_size);

    *engine = eng;
    return GTCRN_OK;
}

extern "C" gtcrn_status_t gtcrn_engine_destroy(gtcrn_engine_t *engine) {
    if (!engine) {
        return GTCRN_ERROR_INVALID;
    }

    /* Delete interpreter and resolver */
    if (engine->interpreter) {
        delete engine->interpreter;
        engine->interpreter = nullptr;
    }

    if (engine->resolver) {
        delete engine->resolver;
        engine->resolver = nullptr;
    }

    /* Free arena if owned */
    if (engine->arena_owned && engine->arena) {
        gtcrn_platform_free(engine->arena);
    }

    /* Free engine structure */
    gtcrn_platform_free(engine);

    GTCRN_LOGI(TAG, "Engine destroyed");
    return GTCRN_OK;
}

/* ============================================================================
 * Model Loading
 * ========================================================================== */

extern "C" gtcrn_status_t gtcrn_engine_load_model(gtcrn_engine_t *engine,
                                                   const uint8_t *model_data,
                                                   size_t model_size) {
    if (!engine || !model_data || model_size == 0) {
        return GTCRN_ERROR_INVALID;
    }

    if (engine->is_ready) {
        GTCRN_LOGW(TAG, "Model already loaded, replacing...");
        /* Clean up existing interpreter */
        if (engine->interpreter) {
            delete engine->interpreter;
            engine->interpreter = nullptr;
        }
        if (engine->resolver) {
            delete engine->resolver;
            engine->resolver = nullptr;
        }
        engine->is_ready = false;
    }

    /* Get model from buffer */
    engine->model = tflite::GetModel(model_data);
    if (!engine->model) {
        GTCRN_LOGE(TAG, "Failed to load model from buffer");
        return GTCRN_ERROR_INVALID;
    }

    /* Check schema version */
    if (engine->model->version() != GTCRN_TFLITE_SCHEMA_VERSION) {
        GTCRN_LOGE(TAG, "Model schema version %u, expected %d",
                   engine->model->version(), GTCRN_TFLITE_SCHEMA_VERSION);
        return GTCRN_ERROR_INVALID;
    }

    /* Create op resolver */
    engine->resolver = new (std::nothrow) tflite::MicroMutableOpResolver<GTCRN_NUM_OPS>();
    if (!engine->resolver) {
        GTCRN_LOGE(TAG, "Failed to create op resolver");
        return GTCRN_ERROR_MEMORY;
    }

    gtcrn_status_t status = register_ops(engine->resolver);
    if (status != GTCRN_OK) {
        GTCRN_LOGE(TAG, "Failed to register ops");
        delete engine->resolver;
        engine->resolver = nullptr;
        return status;
    }

    /* Create interpreter */
    engine->interpreter = new (std::nothrow) tflite::MicroInterpreter(
        engine->model, *engine->resolver, engine->arena, engine->config.arena_size);

    if (!engine->interpreter) {
        GTCRN_LOGE(TAG, "Failed to create interpreter");
        delete engine->resolver;
        engine->resolver = nullptr;
        return GTCRN_ERROR_MEMORY;
    }

    /* Allocate tensors */
    TfLiteStatus alloc_status = engine->interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        GTCRN_LOGE(TAG, "AllocateTensors() failed");
        delete engine->interpreter;
        delete engine->resolver;
        engine->interpreter = nullptr;
        engine->resolver = nullptr;
        return GTCRN_ERROR_MEMORY;
    }

    engine->stats.arena_used_bytes = engine->interpreter->arena_used_bytes();
    engine->is_ready = true;

    GTCRN_LOGI(TAG, "Model loaded successfully (arena used: %zu / %zu bytes)",
               engine->stats.arena_used_bytes, engine->stats.arena_total_bytes);

    return GTCRN_OK;
}

extern "C" bool gtcrn_engine_is_ready(const gtcrn_engine_t *engine) {
    return engine && engine->is_ready;
}

/* ============================================================================
 * Tensor Access
 * ========================================================================== */

extern "C" int gtcrn_engine_input_count(const gtcrn_engine_t *engine) {
    if (!engine || !engine->interpreter) return 0;
    return static_cast<int>(engine->interpreter->inputs_size());
}

extern "C" int gtcrn_engine_output_count(const gtcrn_engine_t *engine) {
    if (!engine || !engine->interpreter) return 0;
    return static_cast<int>(engine->interpreter->outputs_size());
}

extern "C" gtcrn_status_t gtcrn_engine_get_input_info(const gtcrn_engine_t *engine,
                                                       int index,
                                                       gtcrn_tensor_info_t *info) {
    if (!engine || !engine->interpreter || !info) {
        return GTCRN_ERROR_INVALID;
    }

    if (index < 0 || index >= gtcrn_engine_input_count(engine)) {
        return GTCRN_ERROR_INVALID;
    }

    TfLiteTensor *tensor = engine->interpreter->input(index);
    if (!tensor) {
        return GTCRN_ERROR_INVALID;
    }

    fill_tensor_info(tensor, info);
    return GTCRN_OK;
}

extern "C" gtcrn_status_t gtcrn_engine_get_output_info(const gtcrn_engine_t *engine,
                                                        int index,
                                                        gtcrn_tensor_info_t *info) {
    if (!engine || !engine->interpreter || !info) {
        return GTCRN_ERROR_INVALID;
    }

    if (index < 0 || index >= gtcrn_engine_output_count(engine)) {
        return GTCRN_ERROR_INVALID;
    }

    const TfLiteTensor *tensor = engine->interpreter->output(index);
    if (!tensor) {
        return GTCRN_ERROR_INVALID;
    }

    fill_tensor_info(tensor, info);
    return GTCRN_OK;
}

extern "C" gtcrn_status_t gtcrn_engine_set_input(gtcrn_engine_t *engine,
                                                  int index,
                                                  const void *data,
                                                  size_t size) {
    if (!engine || !engine->interpreter || !data) {
        return GTCRN_ERROR_INVALID;
    }

    if (index < 0 || index >= gtcrn_engine_input_count(engine)) {
        return GTCRN_ERROR_INVALID;
    }

    TfLiteTensor *tensor = engine->interpreter->input(index);
    if (!tensor) {
        return GTCRN_ERROR_INVALID;
    }

    if (size != tensor->bytes) {
        GTCRN_LOGE(TAG, "Input size mismatch: got %zu, expected %zu", size, tensor->bytes);
        return GTCRN_ERROR_INVALID;
    }

    std::memcpy(tensor->data.raw, data, size);
    return GTCRN_OK;
}

extern "C" gtcrn_status_t gtcrn_engine_get_output(const gtcrn_engine_t *engine,
                                                   int index,
                                                   void *data,
                                                   size_t size) {
    if (!engine || !engine->interpreter || !data) {
        return GTCRN_ERROR_INVALID;
    }

    if (index < 0 || index >= gtcrn_engine_output_count(engine)) {
        return GTCRN_ERROR_INVALID;
    }

    const TfLiteTensor *tensor = engine->interpreter->output(index);
    if (!tensor) {
        return GTCRN_ERROR_INVALID;
    }

    size_t copy_size = (size < tensor->bytes) ? size : tensor->bytes;
    std::memcpy(data, tensor->data.raw, copy_size);

    return GTCRN_OK;
}

/* ============================================================================
 * Inference
 * ========================================================================== */

extern "C" gtcrn_status_t gtcrn_engine_invoke(gtcrn_engine_t *engine) {
    if (!engine || !engine->interpreter || !engine->is_ready) {
        return GTCRN_ERROR_NOT_INIT;
    }

    int64_t t0 = gtcrn_platform_get_time_us();

    TfLiteStatus invoke_status = engine->interpreter->Invoke();

    int64_t t1 = gtcrn_platform_get_time_us();

    if (invoke_status != kTfLiteOk) {
        GTCRN_LOGE(TAG, "Invoke() failed");
        return GTCRN_ERROR_GENERIC;
    }

    engine->stats.inference_time_us = t1 - t0;
    engine->stats.inference_count++;

    GTCRN_LOGD(TAG, "Inference completed in %lld us",
               (long long)engine->stats.inference_time_us);

    return GTCRN_OK;
}

/* ============================================================================
 * Statistics
 * ========================================================================== */

extern "C" gtcrn_status_t gtcrn_engine_get_stats(const gtcrn_engine_t *engine,
                                                  gtcrn_stats_t *stats) {
    if (!engine || !stats) {
        return GTCRN_ERROR_INVALID;
    }

    *stats = engine->stats;
    return GTCRN_OK;
}

extern "C" gtcrn_status_t gtcrn_engine_reset_stats(gtcrn_engine_t *engine) {
    if (!engine) {
        return GTCRN_ERROR_INVALID;
    }

    engine->stats.inference_time_us = 0;
    engine->stats.inference_count = 0;
    /* Keep arena stats */

    return GTCRN_OK;
}
