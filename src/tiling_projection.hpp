// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>

namespace tiling {
constexpr float kHalfPi = 1.57079632679489661923f;

struct HorizontalMap {
  float value, sine;
};

// Rotate the strip clockwise by 90 degrees: (u,v) = (-screen_y,screen_x).
// Its finite width then spans the portrait panel; its length runs vertically.
inline HorizontalMap map_horizontal(int x, int width, float offset, bool plane) {
  float wx = (x + 0.5f + offset - width * 0.5f) * (2.0f / width);
  return plane ? HorizontalMap{cosf(kHalfPi * wx), sinf(kHalfPi * wx)}
               : HorizontalMap{wx, 0.0f};
}

inline float map_vertical(int y, int width, int height, float offset, bool plane) {
  float wy = (height * 0.5f - y - 0.5f - offset) * (2.0f / width);
  return plane ? expf(-kHalfPi * wy) : wy;
}
}  // namespace tiling
