// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "tiling_kernel.hpp"
#include <algorithm>
#include <cmath>
#include <arm_mve.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace tiling {
namespace {
// a - b * s for a scalar s (MVE has vfmaq_n_f16 but no vfmsq_n_f16).
inline float16x8_t vfmsq_n_f16(float16x8_t a, float16x8_t b, float s) { return vfmaq_n_f16(a, b, -s); }

// 1/x for eight positive lanes: binary16 seed and two binary16 Newton steps.
inline float16x8_t rcp8(float16x8_t x) {
  int16x8_t i = vsubq_s16(vdupq_n_s16(0x7784), vreinterpretq_s16_f16(x));
  float16x8_t y = vreinterpretq_f16_s16(i);
  y = vmulq_f16(y, vfmsq_f16(vdupq_n_f16(2.0f), x, y));  // y * (2 - x y)
  y = vmulq_f16(y, vfmsq_f16(vdupq_n_f16(2.0f), x, y));
  return y;
}

static_assert(kWidth % 8 == 0, "complete eight-lane blocks without tails");

inline float16x8_t aa_pair(const Half* minus, const Half* plus) {
  const uint16_t indices[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  const auto offsets = vld1q_u16(indices);
  return vpselq_f16(vldrhq_gather_shifted_offset_f16(minus, offsets),
                    vldrhq_gather_shifted_offset_f16(plus, offsets), 0x3333);
}

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
GeometryStats geometry_pass(const RenderState& state, uint16_t* g_accum, float time, int bx, int by) {
  const auto& g_settings = state.settings;
  const auto& g_planes = state.planes;
  const auto& g_colors = state.colors;
  const auto& g_texture = state.texture;
  const auto& g_map_x = state.map_x;
  const auto& g_map_y = state.map_y;
  const auto& g_map_sin = state.map_sin;
  GeometryStats stats{};
  stats.vectors = Antialias ? kPixels / 2 : kPixels / 8;
  const Plane* pl = g_planes;
  const float16x8_t one = vdupq_n_f16(1.0f), zero = vdupq_n_f16(0.0f);
  const float16x8_t two = vdupq_n_f16(2.0f);

  // Moebius drift, as the shader: translation (dx, 0), rotation by 5 deg/s.
  float angle_deg = fmodf(time * 5.0f, 360.0f);
  float angle = angle_deg * static_cast<float>(M_PI) / 180.0f;
  float dx = cosf(2.0f * static_cast<float>(M_PI) * time * 0.1f) * 0.5f;
  const float rot_c = cosf(angle), rot_s = sinf(angle);
  const bool animate = g_settings.animation;

  // Edge test: acosh(1 + hdot^2 / hdot(n,n)) <= w  <=>  hdot^2 / hdot(n,n) <= cosh(w) - 1.
  const float edge_threshold = coshf(g_settings.edge_width) - 1.0f;
  // Texture mapping of the shader for a square texture: zoom * 4 * (-latest + 0.5) + (0.6, 0.5), wrapped.
  const float tex_scale = g_settings.zoom * 4.0f;
  const float16x8_t tex_off_x = vdupq_n_f16(tex_scale * 0.5f + 0.6f);
  const float16x8_t tex_off_y = vdupq_n_f16(tex_scale * 0.5f + 0.5f);
  const float16x8_t tex_size = vdupq_n_f16(static_cast<float>(kTextureSize));
  const int16x8_t tex_max = vdupq_n_s16(kTextureSize - 1);

  const int height = g_settings.half ? kFrameHeight / 2 : kFrameHeight;
  for (int y = 0; y < kHeight; ++y) {
    const int iy = std::min(std::max(by + y, 0), height - 1);
    float16x8_t cy = vdupq_n_f16(g_map_y[Antialias ? 1 : 0][iy]);
    if constexpr (Antialias) {
      cy = vpselq_f16(cy, vdupq_n_f16(g_map_y[2][iy]), 0x0f0f);
    }
    for (int x = 0; x < kWidth; x += Antialias ? 2 : 8) {
    const int p = y * kWidth + x;
    // Consume mapping values immediately, without two strip-sized float buffers.
    float16x8_t px;
    if constexpr (Antialias) {
      // Lanes 0..3 are pixel x's 2x2 samples; lanes 4..7 are pixel x+1.
      px = aa_pair(g_map_x[1] + bx + x, g_map_x[2] + bx + x);
    } else {
      px = vld1q_f16(g_map_x[0] + bx + x);
    }
    float16x8_t py = cy;
    if (g_settings.geometry) {
      float16x8_t sn;
      if constexpr (Antialias) {
        sn = aa_pair(g_map_sin[1] + bx + x, g_map_sin[2] + bx + x);
      } else {
        sn = vld1q_f16(g_map_sin[0] + bx + x);
      }
      float16x8_t cx = vmulq_f16(cy, px), ci = vmulq_f16(cy, sn);
      float16x8_t dx = vaddq_n_f16(cx, 1.0f);
      float16x8_t inv = rcp8(vfmaq_f16(vmulq_f16(dx, dx), ci, ci));
      px = vmulq_f16(vfmaq_f16(vmulq_f16(vsubq_n_f16(cx, 1.0f), dx), ci, ci), inv);
      py = vmulq_f16(vmulq_n_f16(ci, 2.0f), inv);
    }

    if (animate) {
      // z' = e^(i angle) (z - b) / (1 - conj(b) z), b = (dx, 0)
      float16x8_t nx = vsubq_n_f16(px, dx), ny = py;
      float16x8_t dnx = vfmsq_n_f16(one, px, dx), dny = vmulq_n_f16(py, -dx);
      float16x8_t inv = rcp8(vfmaq_f16(vmulq_f16(dnx, dnx), dny, dny));
      float16x8_t rx = vmulq_f16(vfmaq_f16(vmulq_f16(nx, dnx), ny, dny), inv);  // (n * conj(d)) / |d|^2
      float16x8_t ry = vmulq_f16(vfmsq_f16(vmulq_f16(ny, dnx), nx, dny), inv);
      px = vfmsq_n_f16(vmulq_n_f16(rx, rot_c), ry, rot_s);
      py = vfmaq_n_f16(vmulq_n_f16(rx, rot_s), ry, rot_c);
    }

    // Inside the unit disk? Outside lanes still run (with the radius clamped
    // so the maths stay finite) and are masked out during shading.
    float16x8_t n = vfmaq_f16(vmulq_f16(px, px), py, py);
    mve_pred16_t inside = vcmpltq_n_f16(n, 1.0f);
    // The portrait panel has large regions outside the disk. Avoid reflections
    // and texture gathers when all eight lanes are background (also for AA).
    if (inside == 0) {
      for (int c = 0; c < 3; ++c) {
        uint16_t* dst = g_accum + c * kPixels + p;
        if constexpr (Antialias) dst[0] = dst[1] = g_colors[3][c] + 128;
        else vstrhq_u16(dst, vdupq_n_u16(g_colors[3][c] + 128));
      }
      continue;
    }
    // Outside lanes are masked out during shading; park them at the origin so
    // they converge at once instead of pinning the vector to the round cap.
    px = vpselq_f16(px, zero, inside);
    py = vpselq_f16(py, zero, inside);
    n = vpselq_f16(n, zero, inside);

    // Homogeneous hyperboloid: postpone division by w=1-r^2. Reflections
    // are linear, so the common scale cancels in the final disk projection.
    // Keeping w also lets the edge test use the same physical distance.
    float16x8_t w = vsubq_f16(one, n);
    float16x8_t hx = vmulq_f16(two, px);
    float16x8_t hy = vmulq_f16(two, py);
    float16x8_t hz = vaddq_f16(one, n);

    // Reflect into the fundamental triangle, counting reflections; stop when
    // no lane moved during a round (the shader's if(hdot < 0) branches, made
    // branch-free with min(d, 0) and predicated counting).
    // Only odd/even is used by shading. A predicate XOR saves a live vector
    // register compared with four integer reflection counters.
    uint32_t parity = 0;
    for (int round = 0; round < g_settings.iterations; ++round) {
      ++stats.rounds;
      // The first normal is (sinh(a), 0, 0): its reflection is x=abs(x).
      uint32_t moved = predicate_bits(vcmpltq_n_f16(hx, 0.0f));
      parity ^= moved;
      hx = vabsq_f16(hx);
      {
        constexpr int i = 1;
        float16x8_t d = vfmsq_n_f16(vfmaq_n_f16(vmulq_n_f16(hx, pl[i].nx), hy, pl[i].ny), hz, pl[i].nz);
        uint32_t neg = predicate_bits(vcmpltq_n_f16(d, 0.0f));
        moved |= neg;
        parity ^= neg;
        float16x8_t t = vmulq_n_f16(vminnmq_f16(d, zero), pl[i].k);  // 2 min(d, 0) / hdot(n, n)
        hx = vfmsq_n_f16(hx, t, pl[i].nx);
        hy = vfmsq_n_f16(hy, t, pl[i].ny);
        hz = vfmsq_n_f16(hz, t, pl[i].nz);
      }
      // All presets have q=4: the third normal is proportional to (-1,1,0).
      // Reflecting when y<x is exactly a swap, without a general dot product.
      uint32_t neg = predicate_bits(vcmpltq_f16(hy, hx));
      moved |= neg;
      parity ^= neg;
      float16x8_t lo = vminnmq_f16(hx, hy);
      hy = vmaxnmq_f16(hx, hy);
      hx = lo;
      if (moved == 0) break;
      if (round + 1 == g_settings.iterations) ++stats.capped_vectors;
    }

    // Distance to the nearest mirror, as hdot^2 / hdot(n, n), min over the three.
    float16x8_t best = vmulq_f16(hx, hx);
    {
      constexpr int i = 1;
      float16x8_t d = vfmsq_n_f16(vfmaq_n_f16(vmulq_n_f16(hx, pl[i].nx), hy, pl[i].ny), hz, pl[i].nz);
      best = vminnmq_f16(best, vmulq_n_f16(vmulq_f16(d, d), pl[i].half_k));
    }
    float16x8_t diagonal = vsubq_f16(hy, hx);
    best = vminnmq_f16(best, vmulq_n_f16(vmulq_f16(diagonal, diagonal), 0.5f));
    mve_pred16_t on_edge = vcmpleq_f16(best, vmulq_n_f16(vmulq_f16(w, w), edge_threshold));

    uint16x8_t offset;
    if constexpr (Textured) {
      // Back to the disk and into texture space: texel index = v * T + u.
      float16x8_t inv_z = rcp8(vaddq_f16(w, hz));
      float16x8_t lx = vmulq_f16(hx, inv_z), ly = vmulq_f16(hy, inv_z);
      float16x8_t tu = vfmsq_n_f16(tex_off_x, lx, tex_scale);  // zoom*4*(-lx) + offset
      float16x8_t tv = vfmsq_n_f16(tex_off_y, ly, tex_scale);
      tu = vsubq_f16(tu, vrndmq_f16(tu));  // wrap: fract
      tv = vsubq_f16(tv, vrndmq_f16(tv));
      int16x8_t iu = vminq_s16(vcvtq_s16_f16(vmulq_f16(tu, tex_size)), tex_max);
      int16x8_t iv = vminq_s16(vcvtq_s16_f16(vmulq_f16(tv, tex_size)), tex_max);
      int16x8_t index = vaddq_s16(vmulq_n_s16(iv, kTextureSize), iu);
      offset = vreinterpretq_u16_s16(index);
    }
    // Preserve the main-branch A/B convention. Shade each sample before AA.
    for (int c = 0; c < 3; ++c) {
      int16x8_t tile = vpselq_s16(vdupq_n_s16(g_colors[0][c]), vdupq_n_s16(g_colors[1][c]), parity);
      int16x8_t col = vaddq_n_s16(tile, 128);
      if constexpr (Textured) {
        int16x8_t tx = vldrbq_gather_offset_s16(g_texture + c * kTexels, offset);
        // q+128 maps to RGB bytes; one rounding after the half-texture blend.
        col = vshrq_n_s16(vaddq_n_s16(vaddq_s16(tx, tile), 257), 1);
      }
      col = vpselq_s16(vdupq_n_s16(g_colors[2][c] + 128), col, on_edge);
      col = vpselq_s16(col, vdupq_n_s16(g_colors[3][c] + 128), inside);
      uint16_t* dst = g_accum + c * kPixels + p;
      if constexpr (Antialias) {
        const auto rgb = vreinterpretq_u16_s16(col);
        dst[0] = (vaddvq_p_u16(rgb, 0x00ff) + 2) >> 2;
        dst[1] = (vaddvq_p_u16(rgb, 0xff00) + 2) >> 2;
      }
      else vstrhq_u16(dst, vreinterpretq_u16_s16(col));
    }
  }
  }
  return stats;
}

} // namespace
GeometryStats render_strip(const RenderState& state, uint16_t* accum, float time, int bx, int by) {
  return state.settings.texture
      ? (state.settings.aa ? geometry_pass<true, true>(state, accum, time, bx, by)
                           : geometry_pass<false, true>(state, accum, time, bx, by))
      : (state.settings.aa ? geometry_pass<true, false>(state, accum, time, bx, by)
                           : geometry_pass<false, false>(state, accum, time, bx, by));
}
} // namespace tiling
