// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Nonblocking after initialization: 1 = new texture, 0 = waiting, negative = error.
int32_t camera_update_texture(int8_t *texture, unsigned size);
void camera_stop(void);
#ifdef __cplusplus
}
#endif
