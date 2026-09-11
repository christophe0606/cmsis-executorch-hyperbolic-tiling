// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Hyperbolic tiling on the Ethos-U85 + Cortex-M55 (Helium), shown on the
// DevKit-E8's 480x800 panel: a port of christophe0606/shader_linux_glsl,
// Christophe Favergeon's GLSL fragment shader that tiles the Poincare disk
// with reflected copies of a camera image, split the way the NPU render
// demo splits a 3D pipeline:
//
//   CPU   the branchy per-pixel geometry, four pixels per Helium vector:
//         Moebius animation of the point, up to 40 rounds of three
//         hyperbolic reflections with early exit once no lane moves, the
//         edge-distance test, the tile parity and the texture coordinate.
//         It writes a "tiling G-buffer" at 240x400: the gathered texel
//         (3 int8 planes), parity, edge and inside masks (bool).
//         The texture lookup is a Helium gather load per plane (an NPU
//         gather via index_select compiled but fetched the wrong texels on
//         the board), so the G-buffer carries the texel, not the index.
//   NPU   "tile": tile / edge / background colouring as masked blends over
//         the texels, a 2x bilinear upscale to 480x800 and the transpose to
//         interleaved RGB888                               (model/model.py)
//   CPU   the backend's output copy goes straight into the back frame
//         buffer (Helium), which the CDC200 scans out at the next vblank.
//
// The original demo is controlled by an LLM through MCP tools; here the
// same settings (symmetry, geometry, animation, edge and background colour,
// texture zoom) are commands on the UART console, so the CMSIS Developer
// Assistant's serial tools, or a terminal, play that role.
//
// The (camera) texture is procedural: the board build has no camera path
// wired up, so a 128x128 pattern that drifts over time stands in for the
// video frames the original samples.

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

#ifdef APP_HAS_DISPLAY
#include "board_display.h"
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

// Shapes from the exported method (model/model.py via model_io.h).
constexpr int32_t kTexelShape[] = MODEL_TILE_INPUT0_SHAPE;    // {1, 3, H, W}
constexpr int32_t kMaskShape[] = MODEL_TILE_INPUT1_SHAPE;     // {1, 1, H, W}
constexpr int32_t kColorShape[] = MODEL_TILE_INPUT4_SHAPE;    // {1, 3, 1, 1}
constexpr int32_t kFrameShape[] = MODEL_TILE_OUTPUT0_SHAPE;   // {1, FH, FW, 3}
constexpr int kTextureSize = 128;
constexpr int kTexels = kTextureSize * kTextureSize;
constexpr int kHeight = kMaskShape[2];
constexpr int kWidth = kMaskShape[3];
constexpr int kPixels = kWidth * kHeight;
static_assert(kTexelShape[2] == kHeight && kTexelShape[3] == kWidth, "texel planes per pixel");
constexpr int kFrameHeight = kFrameShape[1];
constexpr int kFrameWidth = kFrameShape[2];
constexpr size_t kFrameBytes = static_cast<size_t>(kFrameHeight) * kFrameWidth * 3;
constexpr int kUpscale = kFrameWidth / kWidth;
#ifdef APP_HAS_DISPLAY
static_assert(kFrameWidth == APP_DISPLAY_WIDTH && kFrameHeight == APP_DISPLAY_HEIGHT,
              "the exported frame does not match the board's display");
#endif

// Tiling G-buffer (NPU inputs), 576 kB in the DTCM, plus the texel index
// of every pixel for the reference check and the probe command.
alignas(16) int8_t g_texel[3 * kPixels];
alignas(16) int32_t g_index[kPixels] APP_POOL_ATTRIBUTES;
alignas(16) int8_t g_parity[kPixels];
alignas(16) int8_t g_edge[kPixels];
alignas(16) int8_t g_inside[kPixels];
alignas(16) int8_t g_texture[kTexels * 3];
alignas(16) int8_t g_colors[4][3];  // tile A, tile B, edge, background

// Start point of every pixel in the disk model, for the current geometry
// (identity for the disk, the strip map for the plane): 768 kB, bulk SRAM.
alignas(16) float g_start_x[kPixels] APP_POOL_ATTRIBUTES;
alignas(16) float g_start_y[kPixels] APP_POOL_ATTRIBUTES;

// Two RGB888 frame buffers the display controller scans out.
alignas(32) uint8_t g_framebuffer[2][kFrameBytes] APP_FRAMEBUFFER_ATTRIBUTES;

// ---------------------------------------------------------------------------
// Settings: what the original demo's MCP tools change.
// ---------------------------------------------------------------------------
struct Color {
  float r, g, b;
};
struct Settings {
  int symmetry = 0;        // 0: (2,4,5), 1: (2,4,7), 2: (4,4,4) triangle group
  int geometry = 0;        // 0: disk, 1: plane (strip model)
  bool animation = true;   // Moebius drift
  float zoom = 1.0f;       // texture zoom (the presets set it, "zoom" overrides)
  bool zoom_override = false;
  Color edge{0.0f, 0.0f, 0.0f};
  Color background{0.0f, 0.0f, 0.0f};
  Color tile_a{1.0f, 0.0f, 0.0f};
  Color tile_b{0.0f, 0.0f, 1.0f};
  float edge_width = 0.01f;  // hyperbolic distance
};
Settings g_settings;

struct NamedColor {
  const char* name;
  Color color;
};
const NamedColor kColors[] = {
    {"black", {0, 0, 0}},     {"white", {1, 1, 1}},     {"red", {1, 0, 0}},        {"green", {0, 1, 0}},
    {"blue", {0, 0, 1}},      {"yellow", {1, 1, 0}},    {"cyan", {0, 1, 1}},       {"magenta", {1, 0, 1}},
    {"orange", {1, 0.5f, 0}}, {"grey", {0.5f, 0.5f, 0.5f}}, {"navy", {0.05f, 0.08f, 0.2f}},
};

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

// The start point of every pixel: the disk coordinates of the pixel, or for
// the plane geometry the strip map w -> (e^(pi w / 2) - 1) / (e^(pi w / 2) + 1)
// of the shader. Once per geometry change (scalar libm, ~100 ms).
void build_start_points() {
  const float scale = 2.0f / static_cast<float>(std::min(kWidth, kHeight));
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      float wx = (x + 0.5f - kWidth * 0.5f) * scale;
      float wy = (kHeight * 0.5f - (y + 0.5f)) * scale;  // rows run top-down, the shader's y runs up
      int p = y * kWidth + x;
      if (g_settings.geometry == 1) {
        float e = expf(0.5f * static_cast<float>(M_PI) * wx);
        float cx = e * cosf(0.5f * static_cast<float>(M_PI) * wy);
        float cy = e * sinf(0.5f * static_cast<float>(M_PI) * wy);
        // (c - 1) / (c + 1)
        float dx = cx + 1.0f, dy = cy;
        float inv = 1.0f / (dx * dx + dy * dy);
        g_start_x[p] = ((cx - 1.0f) * dx + cy * dy) * inv;
        g_start_y[p] = (cy * dx - (cx - 1.0f) * dy) * inv;
      } else {
        g_start_x[p] = wx;
        g_start_y[p] = wy;
      }
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
  return quantize<int8_t>(v, 1.0f / MODEL_TILE_INPUT0_SCALE, MODEL_TILE_INPUT0_ZERO_POINT,
                          MODEL_TILE_INPUT0_QMIN, MODEL_TILE_INPUT0_QMAX);
}
// The three masks are bool tensors: one byte per pixel, 0 or 1.
constexpr int8_t kMaskZero = 0, kMaskOne = 1;

void quantize_colors() {
  const Color* c[4] = {&g_settings.tile_a, &g_settings.tile_b, &g_settings.edge, &g_settings.background};
  const float scales[4] = {MODEL_TILE_INPUT4_SCALE, MODEL_TILE_INPUT5_SCALE, MODEL_TILE_INPUT6_SCALE, MODEL_TILE_INPUT7_SCALE};
  const int zps[4] = {MODEL_TILE_INPUT4_ZERO_POINT, MODEL_TILE_INPUT5_ZERO_POINT, MODEL_TILE_INPUT6_ZERO_POINT, MODEL_TILE_INPUT7_ZERO_POINT};
  for (int i = 0; i < 4; ++i) {
    g_colors[i][0] = quantize<int8_t>(c[i]->r, 1.0f / scales[i], zps[i], -128, 127);
    g_colors[i][1] = quantize<int8_t>(c[i]->g, 1.0f / scales[i], zps[i], -128, 127);
    g_colors[i][2] = quantize<int8_t>(c[i]->b, 1.0f / scales[i], zps[i], -128, 127);
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
inline float us(uint32_t c) { return static_cast<float>(c) * 1.0e6f / static_cast<float>(SystemCoreClock); }

uint32_t g_io_copy_cycles = 0;
const uint8_t* g_last_frame = nullptr;  // the frame most recently presented
uint8_t* g_frame_target = nullptr;
uint32_t g_frame_target_free_after = 0;
uint32_t g_vsync_wait_cycles = 0;

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
  int inside_pixels;
};

constexpr int kMaxRounds = 40;  // NUM_ITERATIONS in the shader

GeometryStats geometry_pass(float time) {
  GeometryStats stats{0, 0, 0};
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

  for (int p = 0; p < kPixels; p += 4) {
    ++stats.vectors;
    float32x4_t px = vld1q_f32(g_start_x + p);
    float32x4_t py = vld1q_f32(g_start_y + p);

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
    // so the maths stay finite) and are masked out in the NPU.
    float32x4_t n = vfmaq_f32(vmulq_f32(px, px), py, py);
    mve_pred16_t inside = vcmpltq_n_f32(n, 1.0f);
    stats.inside_pixels += __builtin_popcount(inside & 0x1111);
    // Outside lanes are masked out by the NPU; park them at the origin so
    // they converge at once instead of pinning the vector to the round cap.
    px = vpselq_f32(px, zero, inside);
    py = vpselq_f32(py, zero, inside);
    n = vpselq_f32(n, zero, inside);

    // To the hyperboloid: (2 p / (1 - n), (1 + n) / (1 - n)).
    float32x4_t inv = rcp4(vsubq_f32(one, n));
    float32x4_t hx = vmulq_f32(vmulq_f32(two, px), inv);
    float32x4_t hy = vmulq_f32(vmulq_f32(two, py), inv);
    float32x4_t hz = vmulq_f32(vaddq_f32(one, n), inv);

    // Reflect into the fundamental triangle, counting reflections; stop when
    // no lane moved during a round (the shader's if(hdot < 0) branches, made
    // branch-free with min(d, 0) and predicated counting).
    int32x4_t count = vdupq_n_s32(0);
    for (int round = 0; round < kMaxRounds; ++round) {
      ++stats.rounds;
      mve_pred16_t moved = 0;
      for (int i = 0; i < 3; ++i) {
        float32x4_t d = vfmsq_n_f32(vfmaq_n_f32(vmulq_n_f32(hx, pl[i].nx), hy, pl[i].ny), hz, pl[i].nz);
        mve_pred16_t neg = vcmpltq_n_f32(d, 0.0f);
        moved |= neg;
        count = vaddq_m_n_s32(count, count, 1, neg);
        float32x4_t t = vmulq_n_f32(vminnmq_f32(d, zero), pl[i].k);  // 2 min(d, 0) / hdot(n, n)
        hx = vfmsq_n_f32(hx, t, pl[i].nx);
        hy = vfmsq_n_f32(hy, t, pl[i].ny);
        hz = vfmsq_n_f32(hz, t, pl[i].nz);
      }
      if (moved == 0) break;
    }

    // Distance to the nearest mirror, as hdot^2 / hdot(n, n), min over the three.
    float32x4_t best = vdupq_n_f32(1.0e30f);
    for (int i = 0; i < 3; ++i) {
      float32x4_t d = vfmsq_n_f32(vfmaq_n_f32(vmulq_n_f32(hx, pl[i].nx), hy, pl[i].ny), hz, pl[i].nz);
      best = vminnmq_f32(best, vmulq_n_f32(vmulq_f32(d, d), pl[i].half_k));
    }
    mve_pred16_t on_edge = vcmpleq_n_f32(best, edge_threshold);

    // Back to the disk and into texture space: texel index = v * T + u.
    float32x4_t inv_z = rcp4(vaddq_f32(one, hz));
    float32x4_t lx = vmulq_f32(hx, inv_z), ly = vmulq_f32(hy, inv_z);
    float32x4_t tu = vfmsq_n_f32(tex_off_x, lx, tex_scale);  // zoom*4*(-lx) + offset
    float32x4_t tv = vfmsq_n_f32(tex_off_y, ly, tex_scale);
    tu = vsubq_f32(tu, vrndmq_f32(tu));  // wrap: fract
    tv = vsubq_f32(tv, vrndmq_f32(tv));
    int32x4_t iu = vminq_s32(vcvtq_s32_f32(vmulq_f32(tu, tex_size)), tex_max);
    int32x4_t iv = vminq_s32(vcvtq_s32_f32(vmulq_f32(tv, tex_size)), tex_max);
    int32x4_t index = vaddq_s32(vmulq_n_s32(iv, kTextureSize), iu);
    vst1q_s32(g_index + p, index);
    // The texture lookup: one gather load per colour plane, four texels at a time.
    uint32x4_t offset = vreinterpretq_u32_s32(index);
    vstrbq_s32(g_texel + p, vldrbq_gather_offset_s32(g_texture, offset));
    vstrbq_s32(g_texel + kPixels + p, vldrbq_gather_offset_s32(g_texture + kTexels, offset));
    vstrbq_s32(g_texel + 2 * kPixels + p, vldrbq_gather_offset_s32(g_texture + 2 * kTexels, offset));

    // Masks as the quantized codes for 1 and 0 (int8 tensors).
    const int32x4_t q_one = vdupq_n_s32(kMaskOne), q_zero = vdupq_n_s32(kMaskZero);
    mve_pred16_t parity = vcmpneq_n_s32(vandq_s32(count, vdupq_n_s32(1)), 0);
    vstrbq_s32(g_parity + p, vpselq_s32(q_one, q_zero, parity));
    vstrbq_s32(g_edge + p, vpselq_s32(q_one, q_zero, on_edge));
    vstrbq_s32(g_inside + p, vpselq_s32(q_one, q_zero, inside));
  }
  return stats;
}

// ---------------------------------------------------------------------------
// Reference of the NPU's composition for a check on a grid of frame pixels.
// ---------------------------------------------------------------------------
void reference_color(int x, int y, float rgb[3]) {
  int p = y * kWidth + x;
  float parity = g_parity[p] ? 1.0f : 0.0f, edge = g_edge[p] ? 1.0f : 0.0f, inside = g_inside[p] ? 1.0f : 0.0f;
  const int8_t tx[3] = {g_texel[p], g_texel[kPixels + p], g_texel[2 * kPixels + p]};
  const Color& a = g_settings.tile_a;
  const Color& b = g_settings.tile_b;
  const float tile[3] = {a.r * parity + b.r * (1 - parity), a.g * parity + b.g * (1 - parity), a.b * parity + b.b * (1 - parity)};
  const float e[3] = {g_settings.edge.r, g_settings.edge.g, g_settings.edge.b};
  const float bg[3] = {g_settings.background.r, g_settings.background.g, g_settings.background.b};
  for (int c = 0; c < 3; ++c) {
    float t = dequantize(tx[c], MODEL_TILE_INPUT0_SCALE, MODEL_TILE_INPUT0_ZERO_POINT);
    float col = 0.5f * t + 0.5f * tile[c];
    col = e[c] * edge + col * (1 - edge);
    rgb[c] = col * inside + bg[c] * (1 - inside);
  }
}

void reference_frame_pixel(int fx, int fy, float rgb[3]) {  // after the bilinear upscale (align_corners=False)
  float sx = std::max((fx + 0.5f) / kUpscale - 0.5f, 0.0f), sy = std::max((fy + 0.5f) / kUpscale - 0.5f, 0.0f);
  int x0 = std::min(static_cast<int>(sx), kWidth - 1), y0 = std::min(static_cast<int>(sy), kHeight - 1);
  int x1 = std::min(x0 + 1, kWidth - 1), y1 = std::min(y0 + 1, kHeight - 1);
  float tx = sx - x0, ty = sy - y0;
  float c00[3], c01[3], c10[3], c11[3];
  reference_color(x0, y0, c00);
  reference_color(x1, y0, c01);
  reference_color(x0, y1, c10);
  reference_color(x1, y1, c11);
  for (int i = 0; i < 3; ++i) {
    rgb[i] = (1 - ty) * ((1 - tx) * c00[i] + tx * c01[i]) + ty * ((1 - tx) * c10[i] + tx * c11[i]);
  }
}

// Diagnostic: a frame pixel, its G-buffer entry, the reference and the NPU's colour.
void print_pixel(const uint8_t* frame, int x, int y) {
  x = std::min(std::max(x, 0), kFrameWidth - 1);
  y = std::min(std::max(y, 0), kFrameHeight - 1);
  int sx = x / kUpscale, sy = y / kUpscale, p = sy * kWidth + sx;
  float ref[3];
  reference_frame_pixel(x, y, ref);
  const uint8_t* px = frame + (static_cast<size_t>(y) * kFrameWidth + x) * 3;
  const int8_t tx[3] = {g_texel[p], g_texel[kPixels + p], g_texel[2 * kPixels + p]};
  printf("  pixel (%d,%d) <- gbuf (%d,%d): index %ld texel %d,%d,%d parity %d edge %d inside %d | ref %.0f,%.0f,%.0f got %u,%u,%u\n",
         x, y, sx, sy, static_cast<long>(g_index[p]), tx[0], tx[1], tx[2], g_parity[p], g_edge[p], g_inside[p],
         ref[0] * 255, ref[1] * 255, ref[2] * 255, px[0], px[1], px[2]);
}

float check_frame(const uint8_t* frame) {
  float max_err = 0.0f;
  int worst_x = 0, worst_y = 0;
  for (int gy = 0; gy < 20; ++gy) {
    for (int gx = 0; gx < 12; ++gx) {
      int x = gx * kFrameWidth / 12 + kFrameWidth / 24, y = gy * kFrameHeight / 20 + kFrameHeight / 40;
      float ref[3];
      reference_frame_pixel(x, y, ref);
      const uint8_t* px = frame + (static_cast<size_t>(y) * kFrameWidth + x) * 3;
      for (int c = 0; c < 3; ++c) {
        float got = dequantize(static_cast<int>(px[c]) - 128, MODEL_TILE_OUTPUT0_SCALE, MODEL_TILE_OUTPUT0_ZERO_POINT);
        float err = fabsf(got - ref[c]);
        if (err > max_err) {
          max_err = err;
          worst_x = x;
          worst_y = y;
        }
      }
    }
  }
  if (max_err > 8.0f / 255.0f) print_pixel(frame, worst_x, worst_y);
  return max_err;
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
// Non-blocking receive on the console UART (the retarget's getchar blocks).
int console_getchar_nonblocking() {
  UART_Type* uart = reinterpret_cast<UART_Type*>(UART4_BASE);
  if (uart->UART_LSR & 0x01U) return static_cast<int>(uart->UART_RBR & 0xFFU);
  return -1;
}
#else
int console_getchar_nonblocking() { return -1; }
#endif

bool parse_color(const char* name, Color& out) {
  for (const NamedColor& c : kColors) {
    if (strcmp(name, c.name) == 0) {
      out = c.color;
      return true;
    }
  }
  float r, g, b;
  if (sscanf(name, "%f,%f,%f", &r, &g, &b) == 3) {
    out = {r, g, b};
    return true;
  }
  return false;
}

// Returns a bit mask: 1 = symmetry changed, 2 = geometry changed, 4 = colours changed.
int handle_command(char* line) {
  char* cmd = strtok(line, " \t");
  if (!cmd) return 0;
  char* arg = strtok(nullptr, " \t");
  if (strcmp(cmd, "help") == 0) {
    printf("commands: symmetry 0|1|2 | geometry disk|plane | animation on|off | edge <color> | background <color> | "
           "tile a|b <color> | zoom <f> | reset | status | preview | probe x y  (colours: names or r,g,b)\n");
  } else if (strcmp(cmd, "symmetry") == 0 && arg) {
    g_settings.symmetry = std::min(std::max(atoi(arg), 0), 2);
    printf("symmetry changed\n");
    return 1;
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
    printf("symmetry %d, geometry %s, animation %s, zoom %.2f\n", g_settings.symmetry,
           g_settings.geometry ? "plane" : "disk", g_settings.animation ? "on" : "off", g_settings.zoom);
  } else {
    printf("unknown command; try help\n");
  }
  return 0;
}

int poll_console() {
  static char line[64];
  static int len = 0;
  int changed = 0;
  for (int c; (c = console_getchar_nonblocking()) >= 0;) {
    if (c == '\r' || c == '\n') {
      if (len > 0) {
        line[len] = '\0';
        changed |= handle_command(line);
        len = 0;
      }
    } else if (len < static_cast<int>(sizeof(line)) - 1 && c >= 32) {
      line[len++] = static_cast<char>(c);
    }
  }
  return changed;
}

}  // namespace

// Strong override of the ExecuTorch Ethos-U backend's weak I/O copy hook:
// time every copy, and route the tile method's output, the RGB888 frame,
// into the back frame buffer with Helium, folding in the int8 -> uint8 offset.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size) {
  uint32_t t0 = cycles();
  if (size == kFrameBytes && g_frame_target != nullptr) {
#ifdef APP_HAS_DISPLAY
    uint32_t tw = cycles();
    display_wait_frame(g_frame_target_free_after);
    g_vsync_wait_cycles += cycles() - tw;
#endif
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = g_frame_target;
    const uint8x16_t offset = vdupq_n_u8(0x80);
    for (size_t i = 0; i < size; i += 16) {
      mve_pred16_t pred = vctp8q(static_cast<uint32_t>(size - i));
      vstrbq_p_u8(d + i, veorq_u8(vldrbq_z_u8(s + i, pred), offset), pred);
    }
  } else {
    memcpy(dst, src, size);
  }
  g_io_copy_cycles += cycles() - t0;
}

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();
  cycle_counter_init();

  printf("Hyperbolic tiling: Ethos-U85 gathers, blends and upscales; Cortex-M55 Helium reflects\n");
  printf("  program: %lu bytes, geometry %dx%d, texture %dx%d, frame %dx%d RGB888\n",
         model_pte_size, kWidth, kHeight, kTextureSize, kTextureSize, kFrameWidth, kFrameHeight);

  EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));
  auto names = module.method_names();
  if (!names.ok()) {
    printf("program load failed (err=%u)\n", static_cast<unsigned>(names.error()));
    return 1;
  }

  apply_symmetry();
  build_start_points();
  quantize_colors();
  build_texture();
  update_texture(0.0f);

  TensorBox texel_t(ScalarType::Char, kTexelShape, 4, g_texel);
  TensorBox parity_t(ScalarType::Bool, kMaskShape, 4, g_parity);
  TensorBox edge_t(ScalarType::Bool, kMaskShape, 4, g_edge);
  TensorBox inside_t(ScalarType::Bool, kMaskShape, 4, g_inside);
  TensorBox tile_a_t(ScalarType::Char, kColorShape, 4, g_colors[0]);
  TensorBox tile_b_t(ScalarType::Char, kColorShape, 4, g_colors[1]);
  TensorBox edge_color_t(ScalarType::Char, kColorShape, 4, g_colors[2]);
  TensorBox background_t(ScalarType::Char, kColorShape, 4, g_colors[3]);

  int back = 0;
#ifdef APP_HAS_DISPLAY
  for (int i = 0; i < 2; ++i) {
    memset(g_framebuffer[i], 0, kFrameBytes);
    SCB_CleanDCache_by_Addr(g_framebuffer[i], static_cast<int32_t>(kFrameBytes));
  }
  int32_t dstatus = display_init();
  if (dstatus == 0) dstatus = display_start(g_framebuffer[1]);
  printf("  display: %dx%d RGB888 %s (status %ld)\n", kFrameWidth, kFrameHeight, dstatus == 0 ? "on" : "FAILED", static_cast<long>(dstatus));
  const bool display_on = dstatus == 0;
  const int frame_limit = display_on ? 0 : 8;  // 0: run forever
  const int report_every = display_on ? 120 : 1;
  printf("  console: type 'help' for the commands (the demo's MCP tools)\n");
#else
  const bool display_on = false;
  const int frame_limit = 8;
  const int report_every = 1;
#endif

  uint32_t sum_geometry = 0, sum_texture = 0, sum_tile = 0, sum_tile_copy = 0, sum_vsync = 0, sum_frame = 0;
  uint32_t sum_rounds = 0, sum_vectors = 0;
  int report_frames = 0;
  float worst_error = 0.0f;
  bool pass = true;

  for (int frame = 0; frame_limit == 0 || frame < frame_limit; ++frame) {
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;  // survive a debugger detach
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    uint32_t t_frame = cycles();
    float t = frame * (1.0f / 30.0f);

    // 0. Console: settings changes.
    int changed = poll_console();
    if (changed & 1) apply_symmetry();
    if (changed & 2) build_start_points();
    if (changed & 4) quantize_colors();

    // 1. Texture "video frame" (CPU).
    uint32_t t0 = cycles();
    if (frame % 2 == 0) update_texture(t);
    uint32_t t_texture = cycles() - t0;

    // 2. Geometry pass (CPU, Helium).
    t0 = cycles();
    GeometryStats gs = geometry_pass(t);
    uint32_t t_geometry = cycles() - t0;

    // 3. Compose, upscale (NPU); the output copy lands in the back frame buffer.
    t0 = cycles();
    g_io_copy_cycles = 0;
    g_vsync_wait_cycles = 0;
    g_frame_target = g_framebuffer[back];
    auto out = module.execute(MODEL_TILE_METHOD, {texel_t.evalue(), parity_t.evalue(), edge_t.evalue(),
                                                  inside_t.evalue(), tile_a_t.evalue(), tile_b_t.evalue(),
                                                  edge_color_t.evalue(), background_t.evalue()});
    g_frame_target = nullptr;
    uint32_t t_tile = cycles() - t0;
    uint32_t t_tile_copy = g_io_copy_cycles;
    uint32_t t_vsync = g_vsync_wait_cycles;
    if (!out.ok() || out->empty()) {
      printf("tile method failed (err=%u)\n", static_cast<unsigned>(out.error()));
      return 1;
    }

    // 4. Present.
    const uint8_t* frame_rgb = g_framebuffer[back];
#ifdef APP_HAS_DISPLAY
    if (display_on) {
      SCB_CleanDCache_by_Addr(g_framebuffer[back], static_cast<int32_t>(kFrameBytes));
      display_present(g_framebuffer[back]);
      g_frame_target_free_after = display_frame_count();
      back ^= 1;
    }
#endif
    g_last_frame = frame_rgb;

    // 5. Check on reported frames.
    if ((frame % report_every) == report_every - 1) {
      float max_err = check_frame(frame_rgb);
      worst_error = std::max(worst_error, max_err);
      if (max_err > 8.0f / 255.0f) pass = false;
    }
    uint32_t t_total = cycles() - t_frame;
    sum_geometry += t_geometry; sum_texture += t_texture; sum_tile += t_tile; sum_tile_copy += t_tile_copy;
    sum_vsync += t_vsync; sum_frame += t_total; sum_rounds += gs.rounds; sum_vectors += gs.vectors;
    ++report_frames;

    if ((frame % report_every) == report_every - 1) {
      float n = static_cast<float>(report_frames);
      printf("frames %d..%d avg: geometry %.1f ms (%.1f reflection rounds/vector, %d px inside) | texture %.1f ms | "
             "tile-NPU %.1f ms (frame copy %.1f, vsync wait %.1f) | frame %.1f ms = %.1f fps | max err %.1f/255\n",
             frame - report_frames + 1, frame, us(sum_geometry / n) / 1000.0f,
             static_cast<float>(sum_rounds) / std::max(1u, sum_vectors), gs.inside_pixels, us(sum_texture / n) / 1000.0f,
             us(sum_tile / n) / 1000.0f, us(sum_tile_copy / n) / 1000.0f, us(sum_vsync / n) / 1000.0f,
             us(sum_frame / n) / 1000.0f, 1.0e6f / us(sum_frame / n), worst_error * 255.0f);
      sum_geometry = sum_texture = sum_tile = sum_tile_copy = sum_vsync = sum_frame = sum_rounds = sum_vectors = 0;
      report_frames = 0;
    }
    if (!display_on && frame == frame_limit - 1) {
      printf("last frame, luminance, 48x40 cells:\n");
      ascii_preview(frame_rgb);
    }
  }

  printf("worst NPU-vs-reference error: %.1f of 255\n", worst_error * 255.0f);
  printf("Test_result: %s\n", pass ? "PASS" : "FAIL");
  printf("\x04");
  fflush(stdout);
  return pass ? 0 : 1;
}
