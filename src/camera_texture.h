// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

// Square RGB565 snapshot -> planar RGB int8, with the renderer's q = byte - 128.
// Sample pixel centers so every output texel belongs to the same captured frame.
static inline void camera_rgb565_to_texture(const uint16_t *frame, unsigned width,
                                            int8_t *texture, unsigned size) {
  const unsigned plane = size * size;
  for (unsigned y = 0; y < size; ++y) {
    const unsigned sy = ((2 * y + 1) * width) / (2 * size);
    for (unsigned x = 0; x < size; ++x) {
      const unsigned sx = ((2 * x + 1) * width) / (2 * size);
      const uint16_t pixel = frame[sy * width + sx];
      const unsigned r = (pixel >> 11) & 31, g = (pixel >> 5) & 63, b = pixel & 31;
      const unsigned i = y * size + x;
      texture[i] = (int8_t)((int)((r << 3) | (r >> 2)) - 128);
      texture[plane + i] = (int8_t)((int)((g << 2) | (g >> 4)) - 128);
      texture[2 * plane + i] = (int8_t)((int)((b << 3) | (b >> 2)) - 128);
    }
  }
}
