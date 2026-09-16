// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "tiling_settings.hpp"
#include "tiling_antialiasing.hpp"

namespace tiling {
constexpr int kWidth = 240, kHeight = 16, kPixels = kWidth * kHeight;
constexpr int kFrameWidth = 480, kFrameHeight = 800;
constexpr int kTextureSize = 128, kTexels = kTextureSize * kTextureSize;

struct Plane { float nx, ny, nz, k, half_k; };
struct GeometryStats {
  uint32_t rounds, vectors, capped_vectors;
};

// A frame's immutable inputs. Each core keeps its own copy in local DTCM.
struct alignas(32) RenderState {
  Settings settings;
  Rect aa_center;
  Plane planes[3];
  alignas(16) int8_t colors[4][3];
  alignas(16) int8_t texture[3 * kTexels];
  alignas(16) float map_x[3][kFrameWidth];
  alignas(16) float map_y[3][kFrameHeight];
  alignas(16) float map_sin[3][kFrameWidth];
};

GeometryStats render_strip(const RenderState& state, uint16_t* accum,
                           float time, int bx, int by);
} // namespace tiling
