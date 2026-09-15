// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Helium reflects, shades and averages a 2x2 subpixel grid in bounded strips.
// Full resolution goes directly to RGB888; half resolution uses Ethos to upscale.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "RTE_Components.h"
#include CMSIS_device_header

#include <arm_mve.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "model_io.h"
#include "model_pte.h"
#include "tiling_settings.hpp"
#include "tiling_projection.hpp"
#include "mcp_tools.hpp"
#include "mcp.h"

#ifdef APP_HAS_DISPLAY
#include "board_display.h"
#include "board_console.h"
#endif

#ifndef __ARM_FEATURE_MVE
#error "This runner is written with Helium (MVE) intrinsics; build for a core with MVE (Cortex-M55/M85)."
#endif

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

namespace {

// ---------------------------------------------------------------------------
// Memory: pools, frame buffers, the tiling G-buffer. Placement is
// board-overridable (see the DevKit-E8 layer).
// ---------------------------------------------------------------------------
#ifndef APP_METHOD_POOL_SIZE
#define APP_METHOD_POOL_SIZE (4 * 1024 * 1024)
#endif
#ifndef APP_TEMP_POOL_SIZE
#define APP_TEMP_POOL_SIZE (4 * 1024 * 1024)  // Ethos-U scratch is drawn from here.
#endif
#ifdef APP_POOL_SECTION
#define APP_POOL_ATTRIBUTES __attribute__((section(APP_POOL_SECTION)))
#else
#define APP_POOL_ATTRIBUTES
#endif
#ifdef APP_FRAMEBUFFER_SECTION
#define APP_FRAMEBUFFER_ATTRIBUTES __attribute__((section(APP_FRAMEBUFFER_SECTION)))
#else
#define APP_FRAMEBUFFER_ATTRIBUTES APP_POOL_ATTRIBUTES
#endif

constexpr size_t kMethodPoolSize = APP_METHOD_POOL_SIZE;
constexpr size_t kTempPoolSize = APP_TEMP_POOL_SIZE;
alignas(16) uint8_t g_method_pool[kMethodPoolSize] APP_POOL_ATTRIBUTES;
alignas(16) uint8_t g_temp_pool[kTempPoolSize] APP_POOL_ATTRIBUTES;

constexpr int32_t kRgbShape[] = MODEL_UPSCALE_INPUT0_SHAPE;
constexpr int kWidth = kRgbShape[3], kHeight = kRgbShape[2];
constexpr int kPixels = kWidth * kHeight;
constexpr int kFrameWidth = 480, kFrameHeight = 800;
constexpr size_t kFrameBytes = kFrameWidth * kFrameHeight * 3;
constexpr int kTextureSize = 128, kTexels = kTextureSize * kTextureSize;
static_assert(kWidth == 240 && kHeight == 16, "strip layout");
static_assert(MODEL_UPSCALE_INPUT0_ZERO_POINT == -128 && MODEL_UPSCALE_OUTPUT0_ZERO_POINT == -128, "RGB offset");
static_assert(MODEL_UPSCALE_INPUT0_SCALE > 0.00392156f && MODEL_UPSCALE_INPUT0_SCALE < 0.00392158f &&
              MODEL_UPSCALE_OUTPUT0_SCALE > 0.00392156f && MODEL_UPSCALE_OUTPUT0_SCALE < 0.00392158f, "RGB scale");
#ifdef APP_HAS_DISPLAY
static_assert(kFrameWidth == APP_DISPLAY_WIDTH && kFrameHeight == APP_DISPLAY_HEIGHT, "panel size");
#endif
alignas(16) uint16_t g_accum[3 * kPixels];
alignas(16) int8_t g_rgb[3 * kPixels];
alignas(16) int8_t g_texture[3 * kTexels];
alignas(16) int8_t g_colors[4][3];
// Separable disk/strip mapping tables for all possible subpixel coordinates.
alignas(16) float g_map_x[3][kFrameWidth];
alignas(16) float g_map_y[3][kFrameHeight], g_map_sin[3][kFrameWidth];

// Two RGB888 frame buffers the display controller scans out.
alignas(32) uint8_t g_framebuffer[2][kFrameBytes] APP_FRAMEBUFFER_ATTRIBUTES;

// ---------------------------------------------------------------------------
// Settings: what the original demo's MCP tools change.
// ---------------------------------------------------------------------------
Settings g_settings;

// ---------------------------------------------------------------------------
// Hyperbolic geometry (Minkowski model): the three mirror planes of a (p,q,r)
// triangle group, after the construction in the original demo's
// hyperbolic.cpp. hdot(a, b) = a.x b.x + a.y b.y - a.z b.z.
// ---------------------------------------------------------------------------
struct Vec3 {
  double x, y, z;
};
Vec3 hcross(const Vec3& u, const Vec3& v) {
  return {v.z * u.y - v.y * u.z, -v.z * u.x + v.x * u.z, -(v.y * u.x - v.x * u.y)};
}
double hdot(const Vec3& u, const Vec3& v) { return u.x * v.x + u.y * v.y - u.z * v.z; }

// Mirror normals n1, n2, n3 of the triangle with angles pi/p, pi/q, pi/r,
// via the hyperbolic law of cosines for the side lengths.
void compute_triangle(int p, int q, int r, Vec3& n1, Vec3& n2, Vec3& n3) {
  if (1.0 / p + 1.0 / q + 1.0 / r >= 1.0) {  // not hyperbolic
    p = q = r = 4;
  }
  double alpha = M_PI / p, beta = M_PI / q, gamma = M_PI / r;
  double a = acosh((cos(gamma) * cos(beta) + cos(alpha)) / (sin(gamma) * sin(beta)));
  double b = acosh((cos(gamma) * cos(alpha) + cos(beta)) / (sin(gamma) * sin(alpha)));
  double c = acosh((cos(alpha) * cos(beta) + cos(gamma)) / (sin(alpha) * sin(beta)));
  Vec3 p0{0.0, 0.0, 1.0};
  Vec3 p1{0.0, sinh(a), cosh(a)};
  double u = cosh(c) / tanh(a) - cosh(b) / sinh(a);
  double v = cosh(c);
  Vec3 p2{sqrt(v * v - u * u - 1.0), u, v};
  Vec3 m1 = hcross(p0, p1), m2 = hcross(p1, p2), m3 = hcross(p2, p0);
  n1 = {-m1.x, -m1.y, -m1.z};
  n2 = {-m2.x, -m2.y, -m2.z};
  n3 = {-m3.x, -m3.y, -m3.z};
}

struct Plane {
  float nx, ny, nz;  // mirror normal
  float k;           // 2 / hdot(n, n): reflection step
  float half_k;      // 1 / hdot(n, n): edge distance measure
};
Plane g_planes[3];

void apply_symmetry() {
  static const int presets[3][3] = {{2, 4, 5}, {2, 4, 7}, {4, 4, 4}};
  static const float preset_zoom[3] = {1.0f, 0.5f, 0.5f};
  const int* pqr = presets[std::min(std::max(g_settings.symmetry, 0), 2)];
  Vec3 n[3];
  compute_triangle(pqr[0], pqr[1], pqr[2], n[0], n[1], n[2]);
  for (int i = 0; i < 3; ++i) {
    double nn = hdot(n[i], n[i]);
    g_planes[i] = {static_cast<float>(n[i].x), static_cast<float>(n[i].y), static_cast<float>(n[i].z),
                   static_cast<float>(2.0 / nn), static_cast<float>(1.0 / nn)};
  }
  if (!g_settings.zoom_override) g_settings.zoom = preset_zoom[g_settings.symmetry];
}

inline float32x4_t rcp4(float32x4_t x);

void build_maps() {
  int width = g_settings.half ? kFrameWidth / 2 : kFrameWidth;
  int height = g_settings.half ? kFrameHeight / 2 : kFrameHeight;
  const float offsets[] = {0, -0.25f, 0.25f};
  for (int sample = 0; sample < 3; ++sample) {
    for (int x = 0; x < width; ++x) {
      auto horizontal = tiling::map_horizontal(x, width, offsets[sample], g_settings.geometry != 0);
      g_map_x[sample][x] = horizontal.value;
      g_map_sin[sample][x] = horizontal.sine;
    }
    for (int y = 0; y < height; ++y) {
      g_map_y[sample][y] = tiling::map_vertical(y, width, height, offsets[sample], g_settings.geometry != 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Quantization (per model_io.h) and the texture.
// ---------------------------------------------------------------------------
template <typename T>
inline T quantize(float x, float inv_scale, int zero_point, int qmin, int qmax) {
  constexpr float kBias = 65536.0f;
  int q = static_cast<int>(x * inv_scale + 0.5f + kBias) - 65536 + zero_point;
  return static_cast<T>(std::min(std::max(q, qmin), qmax));
}
inline float dequantize(int q, float scale, int zero_point) {
  return static_cast<float>(q - zero_point) * scale;
}
inline int8_t q_texture(float v) {
  return quantize<int8_t>(v, 1.0f / MODEL_UPSCALE_INPUT0_SCALE, MODEL_UPSCALE_INPUT0_ZERO_POINT,
                          MODEL_UPSCALE_INPUT0_QMIN, MODEL_UPSCALE_INPUT0_QMAX);
}

void quantize_colors() {
  const Color* c[] = {&g_settings.tile_a, &g_settings.tile_b, &g_settings.edge, &g_settings.background};
  for (int i = 0; i < 4; ++i) {
    g_colors[i][0] = q_texture(c[i]->r);
    g_colors[i][1] = q_texture(c[i]->g);
    g_colors[i][2] = q_texture(c[i]->b);
  }
}

// A procedural stand-in for the camera frame: soft colour bands, a ring and
// a checker patch, built once; per frame it scrolls (a row and column
// rotation) so the tiles visibly "play video".
alignas(16) int8_t g_texture_base[kTexels * 3];

void build_texture() {
  constexpr int T = kTextureSize;
  for (int y = 0; y < T; ++y) {
    for (int x = 0; x < T; ++x) {
      float u = (x + 0.5f) / T, v = (y + 0.5f) / T;
      float r = 0.5f + 0.5f * sinf(6.2832f * u);
      float g = 0.5f + 0.5f * sinf(6.2832f * v + 2.0f);
      float b = 0.5f + 0.5f * sinf(6.2832f * (u + v) * 0.5f);
      float dx = u - 0.5f, dy = v - 0.5f;
      float d = sqrtf(dx * dx + dy * dy);
      if (d > 0.16f && d < 0.22f) { r = 1.0f; g = 1.0f; b = 0.9f; }  // bright ring
      if (((x / 16) + (y / 16)) % 2 == 0 && u > 0.7f && v > 0.7f) { r *= 0.3f; g *= 0.3f; b *= 0.3f; }  // checker corner
      g_texture_base[0 * kTexels + y * T + x] = q_texture(r);
      g_texture_base[1 * kTexels + y * T + x] = q_texture(g);
      g_texture_base[2 * kTexels + y * T + x] = q_texture(b);
    }
  }
}

void update_texture(float t) {
  constexpr int T = kTextureSize;
  int shift_y = static_cast<int>(t * 9.0f) % T, shift_x = static_cast<int>(t * 13.0f) % T;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < T; ++y) {
      const int8_t* src = &g_texture_base[c * kTexels + ((y + shift_y) % T) * T];
      int8_t* dst = &g_texture[c * kTexels + y * T];
      memcpy(dst, src + shift_x, T - shift_x);
      memcpy(dst + (T - shift_x), src, shift_x);
    }
  }
}

// ---------------------------------------------------------------------------
// Cycle counter, timing of the backend's copies, the frame-buffer routing.
// ---------------------------------------------------------------------------
void cycle_counter_init() {
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
inline uint32_t cycles() { return DWT->CYCCNT; }
inline float us(uint64_t c) { return static_cast<float>(c) * 1.0e6f / static_cast<float>(SystemCoreClock); }

const uint8_t* g_last_frame = nullptr;  // the frame most recently presented
uint32_t g_frame_target_free_after = 0;

struct TensorBox {
  std::array<SizesType, 4> sizes;
  std::array<DimOrderType, 4> dim_order;
  std::unique_ptr<TensorImpl> impl;
  TensorBox(ScalarType type, const int32_t* shape, int ndim, void* data) {
    for (int i = 0; i < ndim; ++i) {
      sizes[i] = shape[i];
      dim_order[i] = static_cast<DimOrderType>(i);
    }
    impl = std::make_unique<TensorImpl>(type, ndim, sizes.data(), data, dim_order.data());
  }
  EValue evalue() { return EValue(Tensor(impl.get())); }
};

// ---------------------------------------------------------------------------
// The geometry pass (Helium, four pixels per vector).
// ---------------------------------------------------------------------------
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

struct GeometryStats {
  uint32_t rounds;   // reflection rounds executed, summed over vectors
  uint32_t vectors;  // vectors processed
  uint32_t capped_vectors;
};

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
GeometryStats geometry_pass(float time, int bx, int by) {
  GeometryStats stats{};
  stats.vectors = Antialias ? kPixels : kPixels / 4;
  const Plane* pl = g_planes;
  const float32x4_t one = vdupq_n_f32(1.0f), zero = vdupq_n_f32(0.0f);
  const float32x4_t two = vdupq_n_f32(2.0f);

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
  const float32x4_t tex_off_x = vdupq_n_f32(tex_scale * 0.5f + 0.6f);
  const float32x4_t tex_off_y = vdupq_n_f32(tex_scale * 0.5f + 0.5f);
  const float32x4_t tex_size = vdupq_n_f32(static_cast<float>(kTextureSize));
  const int32x4_t tex_max = vdupq_n_s32(kTextureSize - 1);

  const int height = g_settings.half ? kFrameHeight / 2 : kFrameHeight;
  for (int y = 0; y < kHeight; ++y) {
    const int iy = std::min(std::max(by + y, 0), height - 1);
    float32x4_t cy = vdupq_n_f32(g_map_y[Antialias ? 1 : 0][iy]);
    if constexpr (Antialias) {
      cy = vpselq_f32(cy, vdupq_n_f32(g_map_y[2][iy]), 0x00ff);
    }
    for (int x = 0; x < kWidth; x += Antialias ? 1 : 4) {
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

void print_pixel(const uint8_t* frame, int x, int y) {
  x = std::min(std::max(x, 0), kFrameWidth - 1);
  y = std::min(std::max(y, 0), kFrameHeight - 1);
  const uint8_t* p = frame + (y * kFrameWidth + x) * 3;
  printf("pixel (%d,%d): %u,%u,%u\n", x, y, p[0], p[1], p[2]);
}

void ascii_preview(const uint8_t* frame) {
  static const char ramp[] = " .:-=+*#%@";
  constexpr int cols = 48, rows = 40;
  constexpr int cw = kFrameWidth / cols, ch = kFrameHeight / rows;
  for (int r = 0; r < rows; ++r) {
    char line[cols + 1];
    for (int c = 0; c < cols; ++c) {
      float lum = 0.0f;
      for (int y = r * ch; y < (r + 1) * ch; ++y)
        for (int x = c * cw; x < (c + 1) * cw; ++x) {
          const uint8_t* px = frame + (static_cast<size_t>(y) * kFrameWidth + x) * 3;
          lum += (0.30f * px[0] + 0.59f * px[1] + 0.11f * px[2]) / 255.0f;
        }
      line[c] = ramp[std::min(9, std::max(0, static_cast<int>(lum / (cw * ch) * 9.99f)))];
    }
    line[cols] = '\0';
    printf("|%s|\n", line);
  }
}

// ---------------------------------------------------------------------------
// Console commands: the demo's MCP tools, over the UART.
// ---------------------------------------------------------------------------
#ifdef APP_HAS_DISPLAY
// The interrupt handler owns UART RX; foreground only consumes its ring.
int console_getchar_nonblocking() {
  return board_console_getchar();
}
#else
int console_getchar_nonblocking() { return -1; }
#endif

// Returns a bit mask: 1 = symmetry changed, 2 = geometry changed, 4 = colours changed.
int handle_command(char* line) {
  char* cmd = strtok(line, " \t");
  if (!cmd) return 0;
  char* arg = strtok(nullptr, " \t");
  if (strcmp(cmd, "help") == 0) {
    printf("commands: scale full|half | aa on|off | iterations 1..40 | symmetry 0|1|2 | geometry disk|plane | animation on|off | edge <color> | background <color> | "
           "tile a|b <color> | texture on|off | zoom <f> | reset | status | preview | probe x y  (colours: names or r,g,b)\n");
  } else if (strcmp(cmd, "symmetry") == 0 && arg) {
    g_settings.symmetry = std::min(std::max(atoi(arg), 0), 2);
    printf("symmetry changed\n");
    return 1;
  } else if (strcmp(cmd, "scale") == 0 && arg) {
    if (strcmp(arg, "half") && strcmp(arg, "full")) { printf("use full|half\n"); return 0; }
    g_settings.half = strcmp(arg, "half") == 0;
    printf("scale %s\n", arg);
    return 2;
  } else if (strcmp(cmd, "aa") == 0 && arg) {
    if (strcmp(arg, "on") && strcmp(arg, "off")) { printf("use on|off\n"); return 0; }
    g_settings.aa = strcmp(arg, "on") == 0;
    printf("AA %s (2x2 grid)\n", arg);
  } else if (strcmp(cmd, "texture") == 0 && arg) {
    if (strcmp(arg, "on") && strcmp(arg, "off")) { printf("use on|off\n"); return 0; }
    g_settings.texture = strcmp(arg, "on") == 0;
    printf("texture %s\n", arg);
  } else if (strcmp(cmd, "iterations") == 0 && arg) {
    char* end;
    long n = strtol(arg, &end, 10);
    if (*end || n < 1 || n > kMaxRounds) { printf("iterations must be 1..40\n"); return 0; }
    g_settings.iterations = static_cast<int>(n);
    printf("iterations %d\n", g_settings.iterations);
  } else if (strcmp(cmd, "geometry") == 0 && arg) {
    g_settings.geometry = strcmp(arg, "plane") == 0 ? 1 : 0;
    printf("geometry changed\n");
    return 2;
  } else if (strcmp(cmd, "animation") == 0 && arg) {
    g_settings.animation = strcmp(arg, "on") == 0 || strcmp(arg, "1") == 0;
    printf("animation %s\n", g_settings.animation ? "started" : "stopped");
  } else if ((strcmp(cmd, "edge") == 0 || strcmp(cmd, "background") == 0) && arg) {
    Color c;
    if (!parse_color(arg, c)) {
      printf("unknown colour %s\n", arg);
      return 0;
    }
    (strcmp(cmd, "edge") == 0 ? g_settings.edge : g_settings.background) = c;
    printf("%s colour changed\n", cmd);
    return 4;
  } else if (strcmp(cmd, "tile") == 0 && arg) {
    char* col = strtok(nullptr, " \t");
    Color c;
    if (!col || !parse_color(col, c)) {
      printf("usage: tile a|b <color>\n");
      return 0;
    }
    (arg[0] == 'b' ? g_settings.tile_b : g_settings.tile_a) = c;
    printf("tile colour changed\n");
    return 4;
  } else if (strcmp(cmd, "zoom") == 0 && arg) {
    g_settings.zoom = static_cast<float>(atof(arg));
    g_settings.zoom_override = true;
    printf("zoom changed\n");
  } else if (strcmp(cmd, "reset") == 0) {
    g_settings = Settings();
    printf("settings reset\n");
    return 7;
  } else if (strcmp(cmd, "preview") == 0) {
    if (g_last_frame) ascii_preview(g_last_frame);
  } else if (strcmp(cmd, "probe") == 0 && arg) {
    char* ys = strtok(nullptr, " \t");
    if (g_last_frame && ys) print_pixel(g_last_frame, atoi(arg), atoi(ys));
  } else if (strcmp(cmd, "status") == 0) {
    printf("scale %s, AA %s, iterations %d\n", g_settings.half ? "half" : "full", g_settings.aa ? "on" : "off", g_settings.iterations);
    printf("symmetry %d, geometry %s, animation %s, zoom %.2f\n", g_settings.symmetry,
           g_settings.geometry ? "plane" : "disk", g_settings.animation ? "on" : "off", g_settings.zoom);
    printf("texture %s\n", g_settings.texture ? "on" : "off");
  } else {
    printf("unknown command; try help\n");
  }
  return 0;
}

int poll_console() {
  static char line[4096];
  static int len = 0;
  static bool discard = false;
  // Bound foreground work to one complete line per frame. UART RX continues
  // throughout rendering and replies. Settings cannot change within a frame.
  for (int budget = 0; budget < 8192; ++budget) {
    int c = console_getchar_nonblocking();
    if (c == -1) break;
    if (c == -2) { len = 0; discard = true; continue; }
    if (c == '\r' || c == '\n') {
      if (discard) {
        len = 0;
        discard = false;
        printf("input lost or too long; discarded through newline\n");
        return 0;
      }
      if (len > 0) {
        line[len] = '\0';
        len = 0;
        const char* first = line;
        while (*first == ' ' || *first == '\t') ++first;
        if (*first == '{' || *first == '[') {
          dispatch(first, 0);
          return mcp_take_changes();
        }
        return handle_command(line);
      }
    } else if (!discard && (c >= 32 || c == '\t') && len < static_cast<int>(sizeof(line)) - 1) {
      line[len++] = static_cast<char>(c);
    } else {
      discard = true; // Never execute a truncated or corrupted command.
    }
  }
  return 0;
}

}  // namespace

// Let the NPU output populate its tensor; copy only the useful halo-free rows.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size) {
  memcpy(dst, src, size);
}

extern "C" int app_main(void) {
  mcp_tools_init(g_settings);
  executorch::runtime::runtime_init();
  cycle_counter_init();
  printf("Helium tiling: full/half, 2x2 AA, adjustable reflection limit; Ethos half-scale upscale\n");
  EmbeddedModule module(model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));
  TensorBox rgb_t(ScalarType::Char, kRgbShape, 4, g_rgb);
  auto prepared = module.prepare_method(MODEL_UPSCALE_METHOD);
  if (!prepared.ok()) { printf("upscale preparation failed\n"); return 1; }
  auto* upscale = *prepared;
  if (upscale->set_input(rgb_t.evalue(), 0) != executorch::runtime::Error::Ok) {
    printf("upscale input failed\n"); return 1;
  }
  apply_symmetry();
  build_maps();
  quantize_colors();
  build_texture();
  int back = 0;
  bool display_on = false;
#ifdef APP_HAS_DISPLAY
  for (int i = 0; i < 2; ++i) {
    memset(g_framebuffer[i], 0, kFrameBytes);
    SCB_CleanDCache_by_Addr(g_framebuffer[i], static_cast<int32_t>(kFrameBytes));
  }
  int32_t ds = display_init();
  if (ds == 0) ds = display_start(g_framebuffer[1]);
  display_on = ds == 0;
  printf("display %s (%ld); type help for controls\n", display_on ? "on" : "failed", static_cast<long>(ds));
#endif
  float animation_time = 0;
  uint64_t cycles_since_report = 0;
  for (int frame = 0; display_on || frame < 2; ++frame) {
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    int changed = poll_console();
    if (changed & 1) apply_symmetry();
    if (changed & 2) build_maps();
    if (changed & 4) quantize_colors();
    uint32_t texture_start = cycles();
    if (g_settings.texture) update_texture(g_settings.animation ? animation_time : 0);
    uint64_t total_cycles = cycles() - texture_start;
#ifdef APP_HAS_DISPLAY
    if (display_on && frame && display_wait_frame(g_frame_target_free_after) != 0) {
      printf("display vblank timeout\n"); return 1;
    }
#endif
    const int scale = g_settings.half ? 2 : 1;
    const int width = kFrameWidth / scale, height = kFrameHeight / scale;
    const int step = g_settings.half ? kHeight - 2 : kHeight;
    const int halo = g_settings.half ? 1 : 0;
    const int samples = g_settings.aa ? 4 : 1;
    uint64_t geometry_cycles = 0, upscale_cycles = 0;
    uint32_t rounds = 0, vectors = 0, capped = 0;
    for (int by = 0; by < height; by += step) {
      for (int bx = 0; bx < width; bx += kWidth) {
        uint32_t block_start = cycles();
        {
          uint32_t t = cycles();
          auto stats = g_settings.texture
              ? (g_settings.aa ? geometry_pass<true, true>(animation_time, bx, by - halo)
                               : geometry_pass<false, true>(animation_time, bx, by - halo))
              : (g_settings.aa ? geometry_pass<true, false>(animation_time, bx, by - halo)
                               : geometry_pass<false, false>(animation_time, bx, by - halo));
          geometry_cycles += cycles() - t;
          rounds += stats.rounds; vectors += stats.vectors; capped += stats.capped_vectors;
        }
        const int rows = std::min(step, height - by);
        if (g_settings.half) {
          for (int p = 0; p < 3 * kPixels; p += 4) {
            uint32x4_t col = vldrhq_u32(g_accum + p);
            vstrbq_s32(g_rgb + p, vsubq_n_s32(vreinterpretq_s32_u32(col), 128));
          }
          uint32_t t = cycles();
          auto result = upscale->execute();
          upscale_cycles += cycles() - t;
          if (result != executorch::runtime::Error::Ok) { printf("upscale failed: %u\n", static_cast<unsigned>(result)); return 1; }
          const uint8_t* rgb = reinterpret_cast<const uint8_t*>(upscale->get_output(0).toTensor().const_data_ptr<int8_t>());
          for (int y = 0; y < rows * 2; ++y) {
            uint8_t* dst = g_framebuffer[back] + (by * 2 + y) * kFrameWidth * 3;
            const uint8_t* src = rgb + (y + 2) * kFrameWidth * 3;
            for (int i = 0; i < kFrameWidth * 3; i += 16)
              vst1q_u8(dst + i, veorq_u8(vld1q_u8(src + i), vdupq_n_u8(128)));
          }
        } else {
          const uint32_t offsets[4] = {0, 3, 6, 9};
          const uint32x4_t offsets_v = vld1q_u32(offsets);
          for (int y = 0; y < rows; ++y) {
            uint8_t* dst = g_framebuffer[back] + ((by + y) * kFrameWidth + bx) * 3;
            for (int x = 0; x < kWidth; x += 4)
              for (int c = 0; c < 3; ++c) {
                uint32x4_t col = vldrhq_u32(g_accum + c * kPixels + y * kWidth + x);
                vstrbq_scatter_offset_u32(dst + x * 3 + c, offsets_v, col);
              }
          }
        }
        total_cycles += static_cast<uint32_t>(cycles() - block_start);
      }
    }
    g_last_frame = g_framebuffer[back];
#ifdef APP_HAS_DISPLAY
    if (display_on) {
      SCB_CleanDCache_by_Addr(g_framebuffer[back], static_cast<int32_t>(kFrameBytes));
      if (display_present(g_framebuffer[back]) != 0) { printf("present failed\n"); return 1; }
      g_frame_target_free_after = display_frame_count();
      back ^= 1;
    }
#endif
    cycles_since_report += total_cycles;
    if (frame == 0 || cycles_since_report >= SystemCoreClock) {
      printf("frame %d: %s AA %d iter %d | geometry %.1f ms, upscale %.1f ms | render %.1f ms | rounds/vector %.2f, capped %lu/%lu\n",
           frame, g_settings.half ? "half" : "full", samples, g_settings.iterations,
           us(geometry_cycles) / 1000, us(upscale_cycles) / 1000, us(total_cycles) / 1000,
           static_cast<float>(rounds) / vectors, static_cast<unsigned long>(capped), static_cast<unsigned long>(vectors));
      cycles_since_report = 0;
    }
    if (g_settings.animation) animation_time += us(total_cycles) * 1.0e-6f;
  }
  printf("Test_result: PASS\n\x04");
  return 0;
}
