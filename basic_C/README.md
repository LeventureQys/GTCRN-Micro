# GTCRN-Micro SDK

Cross-platform neural network inference library for GTCRN (Gated Temporal Convolutional Recurrent Network) models.

## Features

- **Platform Independent**: Core API abstracts away platform-specific details
- **TensorFlow Lite Micro**: Uses TFLite Micro for inference
- **Multiple Platform Support**:
  - Standard C/C++ (Linux, Windows, macOS)
  - ESP32/ESP32-S3 (via ESP-IDF)
  - Easy to port to other platforms
- **Simple API**: Clean C API with optional C++ features

## Project Structure

```
basic_C/
├── include/
│   └── gtcrn_micro/
│       ├── gtcrn_micro.h      # Main include header
│       ├── gtcrn_config.h     # Configuration options
│       ├── gtcrn_platform.h   # Platform abstraction layer (PAL)
│       └── gtcrn_engine.h     # Inference engine API
├── src/
│   ├── gtcrn_engine.cpp       # Engine implementation
│   ├── gtcrn_version.c        # Version info
│   └── platform/
│       ├── platform_default.c # Standard C/C++ implementation
│       └── platform_esp32.c   # ESP32 implementation
├── examples/
│   ├── simple_inference.cpp   # Desktop example
│   └── esp32_example.c        # ESP32 example
├── cmake/
│   └── gtcrn_micro-config.cmake.in
└── CMakeLists.txt
```

## Quick Start

### Desktop (Linux/Windows/macOS)

```bash
mkdir build && cd build
cmake -DTFLITE_MICRO_PATH=/path/to/tflite-micro ..
cmake --build .
```

### ESP32 (ESP-IDF)

1. Add the SDK to your ESP-IDF project's components
2. Include the TFLite Micro component: `espressif/esp-tflite-micro`
3. Use `platform_esp32.c` as the platform implementation

## Usage Example

```c
#include "gtcrn_micro/gtcrn_micro.h"

int main() {
    // Initialize platform
    gtcrn_platform_init();

    // Create engine
    gtcrn_engine_t *engine;
    gtcrn_engine_config_t config = GTCRN_ENGINE_CONFIG_DEFAULT();
    gtcrn_engine_create(&config, &engine);

    // Load model
    gtcrn_engine_load_model(engine, model_data, model_size);

    // Set input data
    gtcrn_engine_set_input(engine, 0, input_data, input_size);

    // Run inference
    gtcrn_engine_invoke(engine);

    // Get output
    gtcrn_engine_get_output(engine, 0, output_data, output_size);

    // Cleanup
    gtcrn_engine_destroy(engine);
    gtcrn_platform_deinit();

    return 0;
}
```

## Porting to New Platforms

To port this SDK to a new platform:

1. Create a new `platform_xxx.c` file in `src/platform/`
2. Implement all functions declared in `gtcrn_platform.h`:
   - `gtcrn_platform_init/deinit`
   - `gtcrn_platform_malloc/free`
   - `gtcrn_platform_get_time_us/ms`
   - `gtcrn_platform_delay_ms`
   - `gtcrn_platform_log`
   - `gtcrn_platform_enter/exit_critical`

## Dependencies

- **TensorFlow Lite Micro**: Required for inference
  - Desktop: https://github.com/tensorflow/tflite-micro
  - ESP32: `espressif/esp-tflite-micro` component

## License

Apache-2.0
