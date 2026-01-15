# Examples CMakeLists.txt
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

# Simple inference example
add_executable(gtcrn_example_simple
    simple_inference.cpp
)

target_link_libraries(gtcrn_example_simple
    PRIVATE gtcrn_micro
)

# Note: For actual use, you need to:
# 1. Link against TFLite Micro library
# 2. Provide the model data file
