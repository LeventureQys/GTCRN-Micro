# GTCRN-Micro SDK CMake Build System
# Cross-platform neural network inference library
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

# ============================================================================
# Project Configuration
# ============================================================================

project(gtcrn_micro
    VERSION 1.0.0
    DESCRIPTION "GTCRN-Micro: Cross-platform neural network inference SDK"
    LANGUAGES C CXX
)

# C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Options
# ============================================================================

option(GTCRN_BUILD_EXAMPLES "Build example applications" ON)
option(GTCRN_PLATFORM_ESP32 "Build for ESP32 platform (requires ESP-IDF)" OFF)
option(GTCRN_ENABLE_TESTS "Enable unit tests" OFF)

# ============================================================================
# TensorFlow Lite Micro Dependency
# ============================================================================

# Users must provide TFLite Micro path
if(NOT DEFINED TFLITE_MICRO_PATH)
    message(STATUS "TFLITE_MICRO_PATH not set, checking for installed package...")

    # Try to find TFLite Micro via pkg-config or common paths
    find_path(TFLITE_MICRO_INCLUDE_DIR
        NAMES tensorflow/lite/micro/micro_interpreter.h
        PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite-micro
            $ENV{TFLITE_MICRO_PATH}
            /usr/local/include
            /usr/include
    )

    if(TFLITE_MICRO_INCLUDE_DIR)
        set(TFLITE_MICRO_PATH ${TFLITE_MICRO_INCLUDE_DIR})
        message(STATUS "Found TFLite Micro at: ${TFLITE_MICRO_PATH}")
    else()
        message(WARNING
            "TensorFlow Lite Micro not found. Please set TFLITE_MICRO_PATH.\n"
            "You can download it from: https://github.com/tensorflow/tflite-micro\n"
            "Example: cmake -DTFLITE_MICRO_PATH=/path/to/tflite-micro .."
        )
    endif()
endif()

# ============================================================================
# Source Files
# ============================================================================

set(GTCRN_SOURCES
    src/gtcrn_engine.cpp
    src/gtcrn_stream.cpp
    src/gtcrn_model.c
    src/gtcrn_version.c
)

# Platform-specific sources
if(GTCRN_PLATFORM_ESP32)
    list(APPEND GTCRN_SOURCES src/platform/platform_esp32.c)
else()
    list(APPEND GTCRN_SOURCES src/platform/platform_default.c)
endif()

# ============================================================================
# Library Target
# ============================================================================

add_library(gtcrn_micro STATIC ${GTCRN_SOURCES})

target_include_directories(gtcrn_micro
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# TFLite Micro includes
if(TFLITE_MICRO_PATH)
    target_include_directories(gtcrn_micro
        PUBLIC
            $<BUILD_INTERFACE:${TFLITE_MICRO_PATH}>
            $<INSTALL_INTERFACE:include>
    )
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(gtcrn_micro PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(gtcrn_micro PRIVATE pthread)
endif()

# ============================================================================
# Examples
# ============================================================================

if(GTCRN_BUILD_EXAMPLES AND NOT GTCRN_PLATFORM_ESP32)
    add_subdirectory(examples)
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS gtcrn_micro
    EXPORT gtcrn_micro-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT gtcrn_micro-targets
    FILE gtcrn_micro-targets.cmake
    NAMESPACE gtcrn_micro::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gtcrn_micro
)

# ============================================================================
# Package Configuration
# ============================================================================

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gtcrn_micro-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gtcrn_micro-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gtcrn_micro
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gtcrn_micro-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gtcrn_micro-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gtcrn_micro-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gtcrn_micro
)
