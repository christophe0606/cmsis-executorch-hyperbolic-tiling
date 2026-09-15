// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "tiling_projection.hpp"
#include <complex>
#include <cstdio>

int main() {
  // Every pixel/subpixel must lie inside the disk after the portrait strip
  // transform, including top/bottom corners and half-resolution AA samples.
  constexpr double half_pi = 1.57079632679489661923;
  for (int width : {240, 480}) {
    int height = width * 5 / 3;
    for (float ox : {0.0f, -0.25f, 0.25f}) {
      for (float oy : {0.0f, -0.25f, 0.25f}) {
        for (int y = 0; y < height; ++y) {
          float magnitude = tiling::map_vertical(y, width, height, oy, true);
          double sy = (height * 0.5 - y - 0.5 - oy) * 2.0 / width;
          for (int x = 0; x < width; ++x) {
            double sx = (x + 0.5 + ox - width * 0.5) * 2.0 / width;
            auto horizontal = tiling::map_horizontal(x, width, ox, true);
            // Same real arithmetic as the firmware, with exact host division.
            float cx = magnitude * horizontal.value, ci = magnitude * horizontal.sine;
            float dx = cx + 1.0f, inv = 1.0f / (dx * dx + ci * ci);
            std::complex<double> got{((cx - 1.0f) * dx + ci * ci) * inv, 2.0f * ci * inv};
            // Independent complex reference: the original strip map evaluated
            // at the inverse rotation of the screen coordinates.
            auto exponential = std::exp(half_pi * std::complex<double>{-sy, sx});
            auto expected = (exponential - 1.0) / (exponential + 1.0);
            auto disk_x = tiling::map_horizontal(x, width, ox, false);
            auto disk_y = tiling::map_vertical(y, width, height, oy, false);
            if (std::norm(got) >= 1.0 || std::abs(got - expected) > 1e-6 ||
                std::abs(disk_x.value - sx) > 2e-7 || std::abs(disk_y - sy) > 2e-7) {
              std::fprintf(stderr, "Projection failed at %d,%d (%dx%d, offsets %g,%g)\n",
                           x, y, width, height, ox, oy);
              return 1;
            }
          }
        }
      }
    }
  }
  std::puts("Portrait strip covers all full/half resolution samples; disk mapping unchanged.");
}
