// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "tiling_antialiasing.hpp"
#include <cstdio>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); return 1; } } while (0)

int main() {
  constexpr double pi = 3.14159265358979323846;
  const int presets[][3] = {{2, 4, 5}, {2, 4, 7}, {4, 4, 4}};
  for (int width : {240, 480}) for (int geometry : {0, 1}) for (int symmetry : {0, 1, 2}) {
    const int height = width * 5 / 3;
    Settings settings;
    settings.symmetry = symmetry;
    settings.geometry = geometry;
    auto center = tiling::partial_center(settings, width, height);
    CHECK(center.x0 % 4 == 0 && center.x1 % 4 == 0);
    CHECK(center.x0 < center.x1 && center.y0 < center.y1);
    CHECK(center.x0 == width - center.x1 && center.y0 == height - center.y1);
    const auto& p = presets[symmetry];
    const double area = pi * (1 - 1.0 / p[0] - 1.0 / p[1] - 1.0 / p[2]);
    int aa_pixels = 0;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
      const double sx = (x + 0.5 - width * 0.5) * 2 / width;
      const double sy = (height * 0.5 - y - 0.5) * 2 / width;
      const double r2 = sx * sx + sy * sy;
      const bool aa = x < center.x0 || x >= center.x1 || y < center.y0 || y >= center.y1;
      aa_pixels += aa;
      // Independently evaluate the screen-space hyperbolic metric. Every
      // estimated triangle <= 3 pixels and every disk-rim pixel must get AA.
      const double density = geometry ? pi / (width * std::cos(pi * sx / 2))
                                      : 4.0 / (width * (1 - r2));
      if (geometry || r2 < 1) {
        if (area / (density * density) <= 3) CHECK(aa);
      } else CHECK(aa);
    }
    CHECK(aa_pixels > 0 && aa_pixels < width * height);
    // Model the actual scheduler, including half-scale overlapping halos.
    const int step = width == 240 ? 14 : 16, halo = width == 240 ? 1 : 0;
    for (int by = -halo; by < height; by += step) for (int bx = 0; bx < width; bx += 240) {
      const auto c = tiling::strip_center(center, bx, by, 240, 16, height);
      CHECK(c.x0 % 4 == 0 && c.x1 % 4 == 0);
      const tiling::Rect pieces[] = {c, {0, 0, 240, c.y0}, {0, c.y1, 240, 16},
                                    {0, c.y0, c.x0, c.y1}, {c.x1, c.y0, 240, c.y1}};
      for (int y = 0; y < 16; ++y) for (int x = 0; x < 240; ++x) {
        int covered = 0;
        for (auto rect : pieces)
          covered += x >= rect.x0 && x < rect.x1 && y >= rect.y0 && y < rect.y1;
        CHECK(covered == 1);
        const int gy = std::clamp(by + y, 0, height - 1);
        const bool global_center = bx + x >= center.x0 && bx + x < center.x1 && gy >= center.y0 && gy < center.y1;
        const bool local_center = x >= c.x0 && x < c.x1 && y >= c.y0 && y < c.y1;
        CHECK(global_center == local_center);
      }
    }
    std::printf("%dx%d %s symmetry %d: center [%d,%d)-[%d,%d), AA %.1f%%\n",
                width, height, geometry ? "plane" : "disk", symmetry,
                center.x0, center.y0, center.x1, center.y1, 100.0 * aa_pixels / (width * height));
  }
}
