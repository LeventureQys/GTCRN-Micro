/**
 * @file gtcrn_version.c
 * @brief GTCRN-Micro SDK Version Information
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gtcrn_micro/gtcrn_micro.h"

const char *gtcrn_version(void) {
    return GTCRN_MICRO_VERSION_STRING;
}

void gtcrn_version_info(int *major, int *minor, int *patch) {
    if (major) *major = GTCRN_MICRO_VERSION_MAJOR;
    if (minor) *minor = GTCRN_MICRO_VERSION_MINOR;
    if (patch) *patch = GTCRN_MICRO_VERSION_PATCH;
}
