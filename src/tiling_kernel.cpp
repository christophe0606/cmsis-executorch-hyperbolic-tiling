// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "tiling_kernel.hpp"
#include <algorithm>
#include <cmath>
#include <arm_mve.h>
#include "dsp/fast_math_functions.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace tiling {
namespace {
// Only three edge widths are supported. Keep the accurate libm calculation
// at initialization, rather than repeating it for each strip and AA band.
const float kEdgeThresholds[] = {
    coshf(edge_width(EdgeThickness::Thin)) - 1.0f,
    coshf(edge_width(EdgeThickness::Thick)) - 1.0f,
    coshf(edge_width(EdgeThickness::VeryThick)) - 1.0f};

// 1/x for four lanes without a vector divide: bit-hack seed, two Newton steps.
// a - b * s for a scalar s (MVE has vfmaq_n_f32 but no vfmsq_n_f32).
inline float32x4_t vfmsq_n_f32(float32x4_t a, float32x4_t b, float s) { return vfmaq_n_f32(a, b, -s); }

inline float32x4_t rcp4(float32x4_t x) {
  int32x4_t i = vsubq_s32(vdupq_n_s32(0x7EF311C7), vreinterpretq_s32_f32(x));
  float32x4_t y = vreinterpretq_f32_s32(i);
  y = vmulq_f32(y, vfmsq_f32(vdupq_n_f32(2.0f), x, y));  // y * (2 - x y)
  y = vmulq_f32(y, vfmsq_f32(vdupq_n_f32(2.0f), x, y));
  return y;
}

constexpr int kMaxRounds = 40;  // NUM_ITERATIONS in the shader

inline uint32_t predicate_bits(mve_pred16_t predicate) {
  uint32_t bits = predicate;
  // Materialize P0 in a core register now, before the next comparison. Without
  // this constraint AC6 schedules comparisons together and spills P0 to RAM.
  __asm volatile("" : "+r"(bits));
  return bits;
}

// Keep the hot vector kernel separate from the frame scheduler: inlining it
// extends live ranges across the entire frame and causes unnecessary spills.
template <bool Antialias, bool Textured>
__attribute__((noinline))
GeometryStats geometry_pass(const RenderState& state, uint16_t* g_accum, float time, int bx, int by, Rect rect) {
  if (rect.x0 >= rect.x1 || rect.y0 >= rect.y1) return {};
  const auto& g_settings = state.settings;
  const auto& g_planes = state.planes;
  const auto& g_colors = state.colors;
  const auto& g_texture = state.texture;
  const auto& g_map_x = state.map_x;
  const auto& g_map_y = state.map_y;
  const auto& g_map_sin = state.map_sin;
  GeometryStats stats{};
  const int pixels = (rect.x1 - rect.x0) * (rect.y1 - rect.y0);
  stats.vectors = Antialias ? pixels : pixels / 4;
  const Plane* pl = g_planes;
  const float32x4_t one = vdupq_n_f32(1.0f), zero = vdupq_n_f32(0.0f);
  const float32x4_t two = vdupq_n_f32(2.0f);

  // Moebius drift, as the shader: translation (dx, 0), rotation by 5 deg/s.
  float angle_deg = fmodf(time * 5.0f, 360.0f);
  float angle = angle_deg * static_cast<float>(M_PI) / 180.0f;
  // Bound the drift phase before the DSP table lookup, including long runs.
  const float drift_angle = fmodf(time, 10.0f) * (2.0f * static_cast<float>(M_PI) * 0.1f);
  float dx = arm_cos_f32(drift_angle) * 0.5f;
  const float rot_c = arm_cos_f32(angle), rot_s = arm_sin_f32(angle);
  const bool animate = g_settings.animation;

  // Edge test: acosh(1 + hdot^2 / hdot(n,n)) <= w  <=>  hdot^2 / hdot(n,n) <= cosh(w) - 1.
  const float edge_threshold = kEdgeThresholds[
      g_settings.edge_thickness == EdgeThickness::VeryThick ? 2 :
      g_settings.edge_thickness == EdgeThickness::Thick ? 1 : 0];
  // Texture mapping of the shader for a square texture: zoom * 4 * (-latest + 0.5) + (0.6, 0.5), wrapped.
  const float tex_scale = g_settings.zoom * 4.0f;
  const float32x4_t tex_off_x = vdupq_n_f32(tex_scale * 0.5f + 0.6f);
  const float32x4_t tex_off_y = vdupq_n_f32(tex_scale * 0.5f + 0.5f);
  const float32x4_t tex_size = vdupq_n_f32(static_cast<float>(kTextureSize));
  const int32x4_t tex_max = vdupq_n_s32(kTextureSize - 1);

  const int height = g_settings.half ? kFrameHeight / 2 : kFrameHeight;
  for (int y = rect.y0; y < rect.y1; ++y) {
    const int iy = std::min(std::max(by + y, 0), height - 1);
    float32x4_t cy = vdupq_n_f32(g_map_y[Antialias ? 1 : 0][iy]);
    if constexpr (Antialias) {
      cy = vpselq_f32(cy, vdupq_n_f32(g_map_y[2][iy]), 0x00ff);
    }
    for (int x = rect.x0; x < rect.x1; x += Antialias ? 1 : 4) {
    const int p = y * kWidth + x;
    // Consume mapping values immediately, without two strip-sized float buffers.
    float32x4_t px;
    if constexpr (Antialias) {
      // One pixel's complete 2x2 grid: nearby lanes converge together, and
      // their colors can be reduced immediately without four strip passes.
      px = vpselq_f32(vdupq_n_f32(g_map_x[1][bx + x]),
                      vdupq_n_f32(g_map_x[2][bx + x]), 0x0f0f);
    } else {
      px = vld1q_f32(g_map_x[0] + bx + x);
    }
    float32x4_t py = cy;
    if (g_settings.geometry) {
      float32x4_t sn;
      if constexpr (Antialias) {
        sn = vpselq_f32(vdupq_n_f32(g_map_sin[1][bx + x]),
                       vdupq_n_f32(g_map_sin[2][bx + x]), 0x0f0f);
      } else {
        sn = vld1q_f32(g_map_sin[0] + bx + x);
      }
      float32x4_t cx = vmulq_f32(cy, px), ci = vmulq_f32(cy, sn);
      float32x4_t dx = vaddq_n_f32(cx, 1.0f);
      float32x4_t inv = rcp4(vfmaq_f32(vmulq_f32(dx, dx), ci, ci));
      px = vmulq_f32(vfmaq_f32(vmulq_f32(vsubq_n_f32(cx, 1.0f), dx), ci, ci), inv);
      py = vmulq_f32(vmulq_n_f32(ci, 2.0f), inv);
    }

    if (animate) {
      // z' = e^(i angle) (z - b) / (1 - conj(b) z), b = (dx, 0)
      float32x4_t nx = vsubq_n_f32(px, dx), ny = py;
      float32x4_t dnx = vfmsq_n_f32(one, px, dx), dny = vmulq_n_f32(py, -dx);
      float32x4_t inv = rcp4(vfmaq_f32(vmulq_f32(dnx, dnx), dny, dny));
      float32x4_t rx = vmulq_f32(vfmaq_f32(vmulq_f32(nx, dnx), ny, dny), inv);  // (n * conj(d)) / |d|^2
      float32x4_t ry = vmulq_f32(vfmsq_f32(vmulq_f32(ny, dnx), nx, dny), inv);
      px = vfmsq_n_f32(vmulq_n_f32(rx, rot_c), ry, rot_s);
      py = vfmaq_n_f32(vmulq_n_f32(rx, rot_s), ry, rot_c);
    }

    // Inside the unit disk? Outside lanes still run (with the radius clamped
    // so the maths stay finite) and are masked out during shading.
    float32x4_t n = vfmaq_f32(vmulq_f32(px, px), py, py);
    mve_pred16_t inside = vcmpltq_n_f32(n, 1.0f);
    // The portrait panel has large regions outside the disk. Avoid reflections
    // and texture gathers when all four lanes are background (exact, also for AA).
    if (inside == 0) {
      for (int c = 0; c < 3; ++c) {
        uint16_t* dst = g_accum + c * kPixels + p;
        if constexpr (Antialias) *dst = g_colors[3][c] + 128;
        else vstrhq_u32(dst, vdupq_n_u32(g_colors[3][c] + 128));
      }
      continue;
    }
    // Outside lanes are masked out during shading; park them at the origin so
    // they converge at once instead of pinning the vector to the round cap.
    px = vpselq_f32(px, zero, inside);
    py = vpselq_f32(py, zero, inside);
    n = vpselq_f32(n, zero, inside);

    // Homogeneous hyperboloid: postpone division by w=1-r^2. Reflections
    // are linear, so the common scale cancels in the final disk projection.
    // Keeping w also lets the edge test use the same physical distance.
    float32x4_t w = vsubq_f32(one, n);
    float32x4_t hx = vmulq_f32(two, px);
    float32x4_t hy = vmulq_f32(two, py);
    float32x4_t hz = vaddq_f32(one, n);

    // Reflect into the fundamental triangle, counting reflections; stop when
    // no lane moved during a round (the shader's if(hdot < 0) branches, made
    // branch-free with min(d, 0) and predicated counting).
    // Only odd/even is used by shading. A predicate XOR saves a live vector
    // register compared with four integer reflection counters.
    uint32_t parity = 0;
    for (int round = 0; round < g_settings.iterations; ++round) {
      ++stats.rounds;
      // The first normal is (sinh(a), 0, 0): its reflection is x=abs(x).
      uint32_t moved = predicate_bits(vcmpltq_n_f32(hx, 0.0f));
      parity ^= moved;
      hx = vabsq_f32(hx);
      {
        constexpr int i = 1;
        float32x4_t d = vfmsq_n_f32(vfmaq_n_f32(vmulq_n_f32(hx, pl[i].nx), hy, pl[i].ny), hz, pl[i].nz);
        uint32_t neg = predicate_bits(vcmpltq_n_f32(d, 0.0f));
        moved |= neg;
        parity ^= neg;
        float32x4_t t = vmulq_n_f32(vminnmq_f32(d, zero), pl[i].k);  // 2 min(d, 0) / hdot(n, n)
        hx = vfmsq_n_f32(hx, t, pl[i].nx);
        hy = vfmsq_n_f32(hy, t, pl[i].ny);
        hz = vfmsq_n_f32(hz, t, pl[i].nz);
      }
      // All presets have q=4: the third normal is proportional to (-1,1,0).
      // Reflecting when y<x is exactly a swap, without a general dot product.
      uint32_t neg = predicate_bits(vcmpltq_f32(hy, hx));
      moved |= neg;
      parity ^= neg;
      float32x4_t lo = vminnmq_f32(hx, hy);
      hy = vmaxnmq_f32(hx, hy);
      hx = lo;
      if (moved == 0) break;
      if (round + 1 == g_settings.iterations) ++stats.capped_vectors;
    }

    // Distance to the nearest mirror, as hdot^2 / hdot(n, n), min over the three.
    float32x4_t best = vmulq_f32(hx, hx);
    {
      constexpr int i = 1;
      float32x4_t d = vfmsq_n_f32(vfmaq_n_f32(vmulq_n_f32(hx, pl[i].nx), hy, pl[i].ny), hz, pl[i].nz);
      best = vminnmq_f32(best, vmulq_n_f32(vmulq_f32(d, d), pl[i].half_k));
    }
    float32x4_t diagonal = vsubq_f32(hy, hx);
    best = vminnmq_f32(best, vmulq_n_f32(vmulq_f32(diagonal, diagonal), 0.5f));
    mve_pred16_t on_edge = vcmpleq_f32(best, vmulq_n_f32(vmulq_f32(w, w), edge_threshold));

    uint32x4_t offset;
    if constexpr (Textured) {
      // Back to the disk and into texture space: texel index = v * T + u.
      float32x4_t inv_z = rcp4(vaddq_f32(w, hz));
      float32x4_t lx = vmulq_f32(hx, inv_z), ly = vmulq_f32(hy, inv_z);
      float32x4_t tu = vfmsq_n_f32(tex_off_x, lx, tex_scale);  // zoom*4*(-lx) + offset
      float32x4_t tv = vfmsq_n_f32(tex_off_y, ly, tex_scale);
      tu = vsubq_f32(tu, vrndmq_f32(tu));  // wrap: fract
      tv = vsubq_f32(tv, vrndmq_f32(tv));
      int32x4_t iu = vminq_s32(vcvtq_s32_f32(vmulq_f32(tu, tex_size)), tex_max);
      int32x4_t iv = vminq_s32(vcvtq_s32_f32(vmulq_f32(tv, tex_size)), tex_max);
      int32x4_t index = vaddq_s32(vmulq_n_s32(iv, kTextureSize), iu);
      offset = vreinterpretq_u32_s32(index);
    }
    // Preserve the main-branch A/B convention. Shade each sample before AA.
    for (int c = 0; c < 3; ++c) {
      int32x4_t tile = vpselq_s32(vdupq_n_s32(g_colors[0][c]), vdupq_n_s32(g_colors[1][c]), parity);
      int32x4_t col = vaddq_n_s32(tile, 128);
      if constexpr (Textured) {
        int32x4_t tx = vldrbq_gather_offset_s32(g_texture + c * kTexels, offset);
        // q+128 maps to RGB bytes; one rounding after the half-texture blend.
        col = vshrq_n_s32(vaddq_n_s32(vaddq_s32(tx, tile), 257), 1);
      }
      col = vpselq_s32(vdupq_n_s32(g_colors[2][c] + 128), col, on_edge);
      col = vpselq_s32(col, vdupq_n_s32(g_colors[3][c] + 128), inside);
      uint16_t* dst = g_accum + c * kPixels + p;
      if constexpr (Antialias) *dst = (vaddvq_u32(vreinterpretq_u32_s32(col)) + 2) >> 2;
      else vstrhq_u32(dst, vreinterpretq_u32_s32(col));
    }
  }
  }
  return stats;
}

template <bool Textured>
GeometryStats render_coverage(const RenderState& state, uint16_t* accum, float time, int bx, int by) {
  constexpr Rect all{0, 0, kWidth, kHeight};
  if (state.settings.aa == Antialiasing::None)
    return geometry_pass<false, Textured>(state, accum, time, bx, by, all);
  if (state.settings.aa == Antialiasing::Full)
    return geometry_pass<true, Textured>(state, accum, time, bx, by, all);
  const int height = state.settings.half ? kFrameHeight / 2 : kFrameHeight;
  const Rect c = strip_center(state.aa_center, bx, by, kWidth, kHeight, height);
  if (c.x0 == c.x1 || c.y0 == c.y1)
    return geometry_pass<true, Textured>(state, accum, time, bx, by, all);
  GeometryStats stats = geometry_pass<false, Textured>(state, accum, time, bx, by, c);
  // Four disjoint rectangular bands: no circular predicate or mode branch
  // in the hot vector loop, and no pixel is shaded twice.
  const Rect bands[] = {{0, 0, kWidth, c.y0}, {0, c.y1, kWidth, kHeight},
                        {0, c.y0, c.x0, c.y1}, {c.x1, c.y0, kWidth, c.y1}};
  for (Rect band : bands) {
    auto part = geometry_pass<true, Textured>(state, accum, time, bx, by, band);
    stats.rounds += part.rounds;
    stats.vectors += part.vectors;
    stats.capped_vectors += part.capped_vectors;
  }
  return stats;
}
} // namespace
GeometryStats render_strip(const RenderState& state, uint16_t* accum, float time, int bx, int by) {
  return state.settings.texture ? render_coverage<true>(state, accum, time, bx, by)
                                : render_coverage<false>(state, accum, time, bx, by);
}
} // namespace tiling
