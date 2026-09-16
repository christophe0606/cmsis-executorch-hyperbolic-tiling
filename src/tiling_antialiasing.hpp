// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include "tiling_settings.hpp"

namespace tiling {
struct Rect { int x0, y0, x1, y1; };  // half-open pixel bounds
constexpr float kPartialTrianglePixels = 3.0f;

// Local projected area estimate = hyperbolic triangle area / metric density.
// Isometries (including the animation) preserve that density. This estimates
// small-tile coverage, not the exact area of each finite projected triangle.
inline Rect partial_center(const Settings& settings, int width, int height) {
  constexpr float pi = 3.14159265358979323846f;
  constexpr float areas[] = {pi * (1 - 0.5f - 0.25f - 0.2f),
                            pi * (1 - 0.5f - 0.25f - 1.0f / 7), pi * 0.25f};
  const float area = areas[std::clamp(settings.symmetry, 0, 2)];
  float sx, sy;
  if (settings.geometry) {
    // Portrait strip metric: pi/(width*cos(pi*screen_x/2)). The two
    // boundaries are vertical; density is independent of screen_y.
    const float cutoff = std::min(1.0f, pi * std::sqrt(kPartialTrianglePixels / area) / width);
    sx = width / pi * std::acos(cutoff);
    sy = height * 0.5f;
  } else {
    // Disk metric per pixel: 4/(width*(1-r*r)). Inscribe a square in
    // the cutoff circle so every point in the fine-triangle annulus gets AA.
    // Its complement intentionally includes extra AA around the square sides.
    const float r2 = std::max(0.0f, 1 - 4 * std::sqrt(kPartialTrianglePixels / area) / width);
    sx = sy = width * 0.5f * std::sqrt(r2 * 0.5f);
  }
  // Shrink the center to whole four-pixel vectors. Pixel footprints, not
  // just centers, stay inside the estimated coarse region.
  int x0 = (static_cast<int>(std::ceil(width * 0.5f - sx)) + 3) & ~3;
  int x1 = static_cast<int>(std::floor(width * 0.5f + sx)) & ~3;
  return {x0, static_cast<int>(std::ceil(height * 0.5f - sy)),
          std::max(x0, x1), static_cast<int>(std::floor(height * 0.5f + sy))};
}

// Intersect the center with a strip, preserving clamped halo rows at the
// frame edges. Both full-width and half-width strips start on vector bounds.
inline Rect strip_center(Rect center, int bx, int by, int width, int height, int frame_height) {
  return {std::clamp(center.x0 - bx, 0, width),
          center.y0 == 0 ? 0 : std::clamp(center.y0 - by, 0, height),
          std::clamp(center.x1 - bx, 0, width),
          center.y1 == frame_height ? height : std::clamp(center.y1 - by, 0, height)};
}
} // namespace tiling
