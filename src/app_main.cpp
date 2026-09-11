// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// NPU render: a small 3D pipeline that uses the Ethos-U85 as a tensor
// coprocessor for the stages that are plain tensor math, the Cortex-M55 with
// Helium (MVE) for the stage that is not, and the DevKit-E8's 480x800 MIPI
// DSI panel to show the result.
//
//   CPU   build a model-view and projection matrix per object for the frame
//   NPU   "vertex": clip-space positions and view-space normals of every
//         object's (padded) vertex batch, two int16 batched matmuls whose
//         batch dimension is the object                     (model/model.py)
//   CPU   perspective divide (Helium), back-face culling, edge-function
//         rasterization four pixels at a time (Helium) into an int8 planar
//         240x400 G-buffer (normal, albedo, fog) with a 16-bit z-buffer
//   NPU   "shade": deferred Lambert lighting + ambient + depth fog + a 3x3
//         Gaussian post filter, a 2x bilinear upscale to 480x800 and a
//         transpose to interleaved RGB888, int8
//   CPU   the NPU backend's output copy is routed (Helium, with the int8 ->
//         uint8 offset folded in) straight into the back frame buffer, which
//         the CDC200 display controller scans out at the next vertical blank
//
// Both NPU methods take and return quantized tensors; the quantization
// parameters come from the generated model_io.h, so the rasterizer writes
// int8 straight into the G-buffer and no float tensor crosses the boundary.
//
// Without a display (the Corstone-320 FVP target) the demo renders a fixed
// number of frames, checks a grid of output pixels against a float reference
// of the same shading and prints an ASCII preview.
//
// EmbeddedModule (arm_embedded_module.hpp) manages program loading, method
// memory and execution, like ExecuTorch's Module class does on POSIX hosts.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

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
// Memory: pools, frame buffers and the G-buffer. Sizes and placement are
// board-overridable; the DevKit-E8 layer routes APP_POOL_SECTION and
// APP_FRAMEBUFFER_SECTION to the 8 MB bulk SRAM because none of that fits
// the 1 MB DTCM, and keeps the G-buffer in the DTCM.
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

// Static shapes of the exported methods (model/model.py). Everything else is
// derived from model_io.h.
constexpr int32_t kVertexShape[] = MODEL_VERTEX_INPUT0_SHAPE;   // {objects, N, 4}
constexpr int32_t kMatShape[] = MODEL_VERTEX_INPUT2_SHAPE;      // {objects, 4, 4}
constexpr int32_t kGBufferShape[] = MODEL_SHADE_INPUT0_SHAPE;   // {1, 3, H, W}
constexpr int32_t kDepthShape[] = MODEL_SHADE_INPUT2_SHAPE;     // {1, 1, H, W}
constexpr int32_t kFrameShape[] = MODEL_SHADE_OUTPUT0_SHAPE;    // {1, FH, FW, 3}
constexpr int kObjects = kVertexShape[0];
constexpr int kMaxVertices = kVertexShape[1];  // per object
constexpr int kHeight = kGBufferShape[2];
constexpr int kWidth = kGBufferShape[3];
constexpr int kPixels = kWidth * kHeight;
constexpr int kFrameHeight = kFrameShape[1];
constexpr int kFrameWidth = kFrameShape[2];
constexpr size_t kFrameBytes = static_cast<size_t>(kFrameHeight) * kFrameWidth * 3;
#ifdef APP_HAS_DISPLAY
static_assert(kFrameWidth == APP_DISPLAY_WIDTH && kFrameHeight == APP_DISPLAY_HEIGHT,
              "the exported frame does not match the board's display");
#endif

// Vertex-stage tensors (int16, NPU in/out).
alignas(16) int16_t g_pos_q[kObjects * kMaxVertices * 4];  // object-space positions, w = 1
alignas(16) int16_t g_nrm_q[kObjects * kMaxVertices * 4];  // object-space normals, w = 0
alignas(16) int16_t g_mvp_q[kObjects * 16];
alignas(16) int16_t g_mv_q[kObjects * 16];

// G-buffer (int8 planar, NPU inputs), 672 kB in the zero-wait-state DTCM;
// the 16-bit z-buffer does not fit there as well and lives in the (cached)
// bulk SRAM next to the pools.
alignas(16) int8_t g_normal[3 * kPixels];
alignas(16) int8_t g_albedo[3 * kPixels];
alignas(16) int8_t g_depth[kPixels];
alignas(16) uint16_t g_zbuffer[kPixels] APP_POOL_ATTRIBUTES;  // NDC depth in 1/65535

// Two RGB888 frame buffers the display controller scans out.
alignas(32) uint8_t g_framebuffer[2][kFrameBytes] APP_FRAMEBUFFER_ATTRIBUTES;

// Per-frame CPU-side vertex data derived from the NPU output.
struct ScreenVertex {
  float x, y;     // pixels
  float z;        // NDC depth in [0, 1]
  float w;        // view-space depth (clip w), for the fog term
  float nx, ny, nz;  // view-space normal
  bool visible;   // in front of the near plane
};
ScreenVertex g_screen[kObjects * kMaxVertices];

// Linear fog on view-space depth, written to the G-buffer depth channel (the
// shade method's fog factor).
constexpr float kFogStart = 2.0f, kFogEnd = 6.5f;

// ---------------------------------------------------------------------------
// Scene: a checkered torus and a striped sphere orbiting it.
// ---------------------------------------------------------------------------
struct Vertex {
  float x, y, z;
  float nx, ny, nz;
  float r, g, b;  // albedo in [0, 1]
};
struct Mesh {
  int first_vertex;  // into g_vertices / the NPU batch (object * kMaxVertices)
  int vertex_count;
  int first_index;
  int triangle_count;
};

constexpr int kTorusRings = 32, kTorusSegments = 16;  // 512 vertices, 1024 triangles
constexpr int kSphereStacks = 14, kSphereSlices = 32;  // 480 vertices, 896 triangles
constexpr int kMaxTriangles = 2 * kTorusRings * kTorusSegments + 2 * kSphereStacks * kSphereSlices;
static_assert(kTorusRings * kTorusSegments <= kMaxVertices, "torus does not fit the vertex batch");
static_assert((kSphereStacks + 1) * kSphereSlices <= kMaxVertices, "sphere does not fit the vertex batch");
static_assert(kObjects >= 2, "the scene has two objects");

Vertex g_vertices[kObjects * kMaxVertices];
uint16_t g_indices[kMaxTriangles * 3];
Mesh g_meshes[kObjects];

void build_torus(Mesh& mesh, int first_vertex, int first_index, float major, float minor) {
  mesh.first_vertex = first_vertex;
  mesh.vertex_count = kTorusRings * kTorusSegments;
  mesh.first_index = first_index;
  mesh.triangle_count = 2 * kTorusRings * kTorusSegments;
  for (int i = 0; i < kTorusRings; ++i) {
    float u = 2.0f * 3.14159265f * i / kTorusRings;
    float cu = cosf(u), su = sinf(u);
    for (int j = 0; j < kTorusSegments; ++j) {
      float v = 2.0f * 3.14159265f * j / kTorusSegments;
      float cv = cosf(v), sv = sinf(v);
      Vertex& vert = g_vertices[first_vertex + i * kTorusSegments + j];
      vert.x = (major + minor * cv) * cu;
      vert.y = minor * sv;
      vert.z = (major + minor * cv) * su;
      vert.nx = cv * cu;
      vert.ny = sv;
      vert.nz = cv * su;
      bool check = ((i / 4) + (j / 4)) % 2 == 0;
      vert.r = check ? 0.92f : 0.16f;
      vert.g = check ? 0.48f : 0.42f;
      vert.b = check ? 0.10f : 0.90f;
    }
  }
  int t = first_index;
  for (int i = 0; i < kTorusRings; ++i) {
    int i1 = (i + 1) % kTorusRings;
    for (int j = 0; j < kTorusSegments; ++j) {
      int j1 = (j + 1) % kTorusSegments;
      uint16_t a = i * kTorusSegments + j, b = i1 * kTorusSegments + j;
      uint16_t c = i1 * kTorusSegments + j1, d = i * kTorusSegments + j1;
      g_indices[t++] = a; g_indices[t++] = b; g_indices[t++] = c;
      g_indices[t++] = a; g_indices[t++] = c; g_indices[t++] = d;
    }
  }
}

void build_sphere(Mesh& mesh, int first_vertex, int first_index, float radius) {
  mesh.first_vertex = first_vertex;
  mesh.vertex_count = (kSphereStacks + 1) * kSphereSlices;
  mesh.first_index = first_index;
  mesh.triangle_count = 2 * kSphereStacks * kSphereSlices;
  for (int i = 0; i <= kSphereStacks; ++i) {
    float phi = 3.14159265f * i / kSphereStacks;  // 0 at the north pole
    float sp = sinf(phi), cp = cosf(phi);
    for (int j = 0; j < kSphereSlices; ++j) {
      float theta = 2.0f * 3.14159265f * j / kSphereSlices;
      Vertex& vert = g_vertices[first_vertex + i * kSphereSlices + j];
      vert.nx = sp * cosf(theta);
      vert.ny = cp;
      vert.nz = sp * sinf(theta);
      vert.x = radius * vert.nx;
      vert.y = radius * vert.ny;
      vert.z = radius * vert.nz;
      bool stripe = (i / 2) % 2 == 0;
      vert.r = stripe ? 0.95f : 0.20f;
      vert.g = stripe ? 0.90f : 0.75f;
      vert.b = stripe ? 0.85f : 0.55f;
    }
  }
  int t = first_index;
  for (int i = 0; i < kSphereStacks; ++i) {
    for (int j = 0; j < kSphereSlices; ++j) {
      int j1 = (j + 1) % kSphereSlices;
      uint16_t a = i * kSphereSlices + j, b = (i + 1) * kSphereSlices + j;
      uint16_t c = (i + 1) * kSphereSlices + j1, d = i * kSphereSlices + j1;
      g_indices[t++] = a; g_indices[t++] = b; g_indices[t++] = c;
      g_indices[t++] = a; g_indices[t++] = c; g_indices[t++] = d;
    }
  }
}

// ---------------------------------------------------------------------------
// Quantization (affine, per model_io.h: q = round(x / scale) + zp).
// ---------------------------------------------------------------------------
// Round-to-nearest without lroundf or a division: the value is shifted
// positive so that the float-to-int conversion truncates towards the rounded
// result, then shifted back. Valid for |x / scale| < 2^16.
template <typename T>
inline T quantize(float x, float inv_scale, int zero_point, int qmin, int qmax) {
  constexpr float kBias = 65536.0f;
  int q = static_cast<int>(x * inv_scale + 0.5f + kBias) - 65536 + zero_point;
  return static_cast<T>(std::min(std::max(q, qmin), qmax));
}
inline float dequantize(int q, float scale, int zero_point) {
  return static_cast<float>(q - zero_point) * scale;
}

constexpr float kInvNormalScale = 1.0f / MODEL_SHADE_INPUT0_SCALE;
constexpr float kInvAlbedoScale = 1.0f / MODEL_SHADE_INPUT1_SCALE;
constexpr float kInvDepthScale = 1.0f / MODEL_SHADE_INPUT2_SCALE;
inline int8_t q_normal(float n) {
  return quantize<int8_t>(n, kInvNormalScale, MODEL_SHADE_INPUT0_ZERO_POINT,
                          MODEL_SHADE_INPUT0_QMIN, MODEL_SHADE_INPUT0_QMAX);
}
inline int8_t q_albedo(float a) {
  return quantize<int8_t>(a, kInvAlbedoScale, MODEL_SHADE_INPUT1_ZERO_POINT,
                          MODEL_SHADE_INPUT1_QMIN, MODEL_SHADE_INPUT1_QMAX);
}
inline int8_t q_depth(float d) {
  return quantize<int8_t>(d, kInvDepthScale, MODEL_SHADE_INPUT2_ZERO_POINT,
                          MODEL_SHADE_INPUT2_QMIN, MODEL_SHADE_INPUT2_QMAX);
}

// Helium: four floats -> four int8 lanes, round to nearest, offset, clamp,
// narrowing predicated store.
inline void q_store4(int8_t* dst, float32x4_t x, float inv_scale, int zp, int qmin, int qmax, mve_pred16_t p) {
  int32x4_t q = vcvtnq_s32_f32(vmulq_n_f32(x, inv_scale));
  q = vaddq_n_s32(q, zp);
  q = vmaxq_s32(q, vdupq_n_s32(qmin));
  q = vminq_s32(q, vdupq_n_s32(qmax));
  vstrbq_p_s32(dst, q, p);
}

// Helium: 1/sqrt(x) for four lanes without a vector square root or divide
// (MVE has neither): the integer bit-hack seed and one Newton-Raphson step,
// ~0.2 % relative error, well inside the G-buffer's 1/127 normal step.
inline float32x4_t rsqrt4(float32x4_t x) {
  int32x4_t i = vreinterpretq_s32_f32(x);
  i = vsubq_s32(vdupq_n_s32(0x5f3759df), vshrq_n_s32(i, 1));
  float32x4_t y = vreinterpretq_f32_s32(i);
  float32x4_t half_x_y2 = vmulq_f32(vmulq_n_f32(x, 0.5f), vmulq_f32(y, y));
  return vmulq_f32(y, vsubq_f32(vdupq_n_f32(1.5f), half_x_y2));
}

// ---------------------------------------------------------------------------
// Matrices. Column-vector convention on the CPU (v' = M v, M[row][col]); the
// NPU method multiplies row vectors, so it gets the transpose.
// ---------------------------------------------------------------------------
using Mat4 = std::array<float, 16>;  // row-major M[r * 4 + c]

Mat4 mat_identity() {
  Mat4 m{};
  m[0] = m[5] = m[10] = m[15] = 1.0f;
  return m;
}
Mat4 mat_mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
  return r;
}
Mat4 mat_rotate_x(float a) {
  Mat4 m = mat_identity();
  m[5] = cosf(a); m[6] = -sinf(a); m[9] = sinf(a); m[10] = cosf(a);
  return m;
}
Mat4 mat_rotate_y(float a) {
  Mat4 m = mat_identity();
  m[0] = cosf(a); m[2] = sinf(a); m[8] = -sinf(a); m[10] = cosf(a);
  return m;
}
Mat4 mat_rotate_z(float a) {
  Mat4 m = mat_identity();
  m[0] = cosf(a); m[1] = -sinf(a); m[4] = sinf(a); m[5] = cosf(a);
  return m;
}
Mat4 mat_translate(float x, float y, float z) {
  Mat4 m = mat_identity();
  m[3] = x; m[7] = y; m[11] = z;
  return m;
}
// Camera looks down +z; NDC depth z/w in [0, 1] between near and far.
Mat4 mat_perspective(float fov_y, float aspect, float near, float far) {
  float f = 1.0f / tanf(fov_y * 0.5f);
  Mat4 m{};
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far / (far - near);
  m[11] = -far * near / (far - near);
  m[14] = 1.0f;
  return m;
}

void quantize_matrix_transposed(const Mat4& m, int16_t* out, float scale, int zp, int qmin, int qmax) {
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) out[c * 4 + r] = quantize<int16_t>(m[r * 4 + c], 1.0f / scale, zp, qmin, qmax);
}

// ---------------------------------------------------------------------------
// Cycle counter.
// ---------------------------------------------------------------------------
void cycle_counter_init() {
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
inline uint32_t cycles() { return DWT->CYCCNT; }
inline float us(uint32_t c) { return static_cast<float>(c) * 1.0e6f / static_cast<float>(SystemCoreClock); }

// Cycles the Ethos-U backend spends copying method inputs into and outputs
// out of the NPU scratch buffer, accumulated per NPU call: the part of the
// round trip that is memcpy, not NPU compute.
uint32_t g_io_copy_cycles = 0;

// Where the shade method's output copy goes instead of the output tensor: the
// back frame buffer (see arm_ethos_io_memcpy below). nullptr = normal copy.
uint8_t* g_frame_target = nullptr;
uint32_t g_frame_target_free_after = 0;  // display frame count the target is free after
uint32_t g_vsync_wait_cycles = 0;

// ---------------------------------------------------------------------------
// Tensor wrapping without the tensor extension (see create_ai_layer.py).
// ---------------------------------------------------------------------------
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
// Vertex post-processing (Helium): dequantize the NPU's int16 clip position
// and view normal of one vertex per vector, perspective divide, viewport.
// ---------------------------------------------------------------------------
void post_vertex(const int16_t* clip, const int16_t* nview, int count) {
  const float32x4_t half = vdupq_n_f32(0.5f);
  for (int i = 0; i < count; ++i) {
    float32x4_t c = vmulq_n_f32(vcvtq_f32_s32(vldrhq_s32(clip + i * 4)), MODEL_VERTEX_OUTPUT0_SCALE);
    float32x4_t n = vmulq_n_f32(vcvtq_f32_s32(vldrhq_s32(nview + i * 4)), MODEL_VERTEX_OUTPUT1_SCALE);
    ScreenVertex& s = g_screen[i];
    float w = vgetq_lane_f32(c, 3);
    s.visible = w > 1e-3f;
    float inv_w = s.visible ? 1.0f / w : 0.0f;  // the one scalar divide per vertex
    // ndc = c / w; x -> (ndc * 0.5 + 0.5) * W, y -> (0.5 - ndc * 0.5) * H, z -> ndc
    float32x4_t ndc = vmulq_n_f32(c, inv_w);
    float32x4_t xy = vfmaq_f32(half, ndc, half);  // ndc * 0.5 + 0.5
    s.x = vgetq_lane_f32(xy, 0) * kWidth;
    s.y = (1.0f - vgetq_lane_f32(xy, 1)) * kHeight;
    s.z = vgetq_lane_f32(ndc, 2);
    s.w = w;
    s.nx = vgetq_lane_f32(n, 0);
    s.ny = vgetq_lane_f32(n, 1);
    s.nz = vgetq_lane_f32(n, 2);
  }
}

// ---------------------------------------------------------------------------
// Rasterizer (Helium, four pixels per iteration): barycentric weights are
// affine in x and y, so each row starts from the box corner and steps by a
// constant vector; coverage, z-test, attribute interpolation, normal
// renormalization and int8 quantization all run lane-parallel under one
// predicate, and the G-buffer stores are narrowing predicated stores.
// ---------------------------------------------------------------------------
struct RasterStats {
  int triangles_drawn;
  int pixels_covered;
  int pixels_tested;  // bounding-box pixels visited
};

void fill_bytes(void* dst, uint8_t value, size_t bytes) {  // Helium memset, 16 bytes per store
  uint8_t* p = static_cast<uint8_t*>(dst);
  uint8x16_t v = vdupq_n_u8(value);
  for (size_t i = 0; i < bytes; i += 16) {
    mve_pred16_t pred = vctp8q(static_cast<uint32_t>(bytes - i));
    vstrbq_p_u8(p + i, v, pred);
  }
}

void clear_gbuffer() {
  fill_bytes(g_normal, static_cast<uint8_t>(q_normal(0.0f)), sizeof(g_normal));
  fill_bytes(g_albedo, static_cast<uint8_t>(q_albedo(0.0f)), sizeof(g_albedo));
  fill_bytes(g_depth, static_cast<uint8_t>(q_depth(1.0f)), sizeof(g_depth));  // background: full fog
  fill_bytes(g_zbuffer, 0xff, sizeof(g_zbuffer));  // far plane
}

RasterStats rasterize() {
  RasterStats stats{0, 0, 0};
  const float32x4_t lane = vcvtq_f32_u32(vidupq_n_u32(0, 1));  // {0, 1, 2, 3}
  const float32x4_t zero = vdupq_n_f32(0.0f), one = vdupq_n_f32(1.0f);
  const float32x4_t zmax = vdupq_n_f32(65535.0f);
  const float inv_fog_range = 1.0f / (kFogEnd - kFogStart);

  for (int o = 0; o < kObjects; ++o) {
    const Mesh& mesh = g_meshes[o];
    const ScreenVertex* screen = g_screen + mesh.first_vertex;
    const Vertex* verts = g_vertices + mesh.first_vertex;
    for (int t = 0; t < mesh.triangle_count; ++t) {
      const uint16_t* idx = &g_indices[mesh.first_index + t * 3];
      const ScreenVertex& a = screen[idx[0]];
      const ScreenVertex& b = screen[idx[1]];
      const ScreenVertex& c = screen[idx[2]];
      if (!(a.visible && b.visible && c.visible)) continue;

      float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      if (area >= 0.0f) continue;  // back-facing (screen y points down, CCW front faces)

      int x0 = std::max(0, static_cast<int>(floorf(std::min({a.x, b.x, c.x}))));
      int x1 = std::min(kWidth - 1, static_cast<int>(ceilf(std::max({a.x, b.x, c.x}))));
      int y0 = std::max(0, static_cast<int>(floorf(std::min({a.y, b.y, c.y}))));
      int y1 = std::min(kHeight - 1, static_cast<int>(ceilf(std::max({a.y, b.y, c.y}))));
      if (x0 > x1 || y0 > y1) continue;
      ++stats.triangles_drawn;
      stats.pixels_tested += (x1 - x0 + 1) * (y1 - y0 + 1);

      const Vertex& va = verts[idx[0]];
      const Vertex& vb = verts[idx[1]];
      const Vertex& vc = verts[idx[2]];
      float inv_area = 1.0f / area;

      // Barycentric planes: w0/w1 at the box corner and their x/y steps.
      float px0 = x0 + 0.5f, py0 = y0 + 0.5f;
      float w0_corner = ((b.x - px0) * (c.y - py0) - (b.y - py0) * (c.x - px0)) * inv_area;
      float w1_corner = ((c.x - px0) * (a.y - py0) - (c.y - py0) * (a.x - px0)) * inv_area;
      float w0_dx = (b.y - c.y) * inv_area, w0_dy = (c.x - b.x) * inv_area;
      float w1_dx = (c.y - a.y) * inv_area, w1_dy = (a.x - c.x) * inv_area;
      float32x4_t w0_row = vfmaq_n_f32(vdupq_n_f32(w0_corner), lane, w0_dx);  // corner + lane * dx
      float32x4_t w1_row = vfmaq_n_f32(vdupq_n_f32(w1_corner), lane, w1_dx);
      const float w0_step = 4.0f * w0_dx, w1_step = 4.0f * w1_dx;

      // Per-vertex fog factors from view-space depth.
      const float fog_a = std::min(std::max((a.w - kFogStart) * inv_fog_range, 0.0f), 1.0f);
      const float fog_b = std::min(std::max((b.w - kFogStart) * inv_fog_range, 0.0f), 1.0f);
      const float fog_c = std::min(std::max((c.w - kFogStart) * inv_fog_range, 0.0f), 1.0f);

      for (int y = y0; y <= y1; ++y, w0_row = vaddq_n_f32(w0_row, w0_dy), w1_row = vaddq_n_f32(w1_row, w1_dy)) {
        float32x4_t w0 = w0_row, w1 = w1_row;
        int p = y * kWidth + x0;
        for (int x = x0; x <= x1; x += 4, p += 4, w0 = vaddq_n_f32(w0, w0_step), w1 = vaddq_n_f32(w1, w1_step)) {
          mve_pred16_t pred = vctp32q(static_cast<uint32_t>(x1 - x + 1));  // row tail
          float32x4_t w2 = vsubq_f32(vsubq_f32(one, w0), w1);
          pred = vcmpgeq_m_n_f32(w0, 0.0f, pred);  // inside all three edges
          pred = vcmpgeq_m_n_f32(w1, 0.0f, pred);
          pred = vcmpgeq_m_n_f32(w2, 0.0f, pred);
          if (pred == 0) continue;

          // z-test against the 16-bit z-buffer (widening load, narrowing store).
          float32x4_t z = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(w0, a.z), w1, b.z), w2, c.z);
          uint32x4_t zq = vcvtq_u32_f32(vmulq_f32(vminnmq_f32(vmaxnmq_f32(z, zero), one), zmax));
          uint32x4_t zold = vldrhq_z_u32(g_zbuffer + p, pred);
          pred = vcmphiq_m_u32(zold, zq, pred);  // zold > zq: closer than what is there
          if (pred == 0) continue;
          vstrhq_p_u32(g_zbuffer + p, zq, pred);
          stats.pixels_covered += __builtin_popcount(pred & 0x1111);

          // View-space normal: interpolate, renormalize, quantize.
          float32x4_t nx = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(w0, a.nx), w1, b.nx), w2, c.nx);
          float32x4_t ny = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(w0, a.ny), w1, b.ny), w2, c.ny);
          float32x4_t nz = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(w0, a.nz), w1, b.nz), w2, c.nz);
          float32x4_t len2 = vfmaq_f32(vfmaq_f32(vmulq_f32(nx, nx), ny, ny), nz, nz);
          float32x4_t inv_len = rsqrt4(vaddq_n_f32(len2, 1e-12f));
          q_store4(g_normal + p, vmulq_f32(nx, inv_len), kInvNormalScale, MODEL_SHADE_INPUT0_ZERO_POINT,
                   MODEL_SHADE_INPUT0_QMIN, MODEL_SHADE_INPUT0_QMAX, pred);
          q_store4(g_normal + kPixels + p, vmulq_f32(ny, inv_len), kInvNormalScale, MODEL_SHADE_INPUT0_ZERO_POINT,
                   MODEL_SHADE_INPUT0_QMIN, MODEL_SHADE_INPUT0_QMAX, pred);
          q_store4(g_normal + 2 * kPixels + p, vmulq_f32(nz, inv_len), kInvNormalScale, MODEL_SHADE_INPUT0_ZERO_POINT,
                   MODEL_SHADE_INPUT0_QMIN, MODEL_SHADE_INPUT0_QMAX, pred);

          // Albedo.
          float32x4_t r = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(w0, va.r), w1, vb.r), w2, vc.r);
          float32x4_t g = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(w0, va.g), w1, vb.g), w2, vc.g);
          float32x4_t bl = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(w0, va.b), w1, vb.b), w2, vc.b);
          q_store4(g_albedo + p, r, kInvAlbedoScale, MODEL_SHADE_INPUT1_ZERO_POINT,
                   MODEL_SHADE_INPUT1_QMIN, MODEL_SHADE_INPUT1_QMAX, pred);
          q_store4(g_albedo + kPixels + p, g, kInvAlbedoScale, MODEL_SHADE_INPUT1_ZERO_POINT,
                   MODEL_SHADE_INPUT1_QMIN, MODEL_SHADE_INPUT1_QMAX, pred);
          q_store4(g_albedo + 2 * kPixels + p, bl, kInvAlbedoScale, MODEL_SHADE_INPUT1_ZERO_POINT,
                   MODEL_SHADE_INPUT1_QMIN, MODEL_SHADE_INPUT1_QMAX, pred);

          // Fog factor.
          float32x4_t fog = vfmaq_n_f32(vfmaq_n_f32(vmulq_n_f32(w0, fog_a), w1, fog_b), w2, fog_c);
          q_store4(g_depth + p, fog, kInvDepthScale, MODEL_SHADE_INPUT2_ZERO_POINT,
                   MODEL_SHADE_INPUT2_QMIN, MODEL_SHADE_INPUT2_QMAX, pred);
        }
      }
    }
  }
  return stats;
}

// ---------------------------------------------------------------------------
// Reference shading, mirroring model/model.py in float from the quantized
// G-buffer, for a check of the NPU output on a grid of frame pixels.
// ---------------------------------------------------------------------------
constexpr float kLightDir[3] = {0.30f, 0.50f, -0.81f};  // = LIGHT_DIR in model/model.py
constexpr float kKeyLight = 0.8f, kAmbient = 0.2f;
constexpr float kFogColor[3] = {0.10f, 0.12f, 0.18f};
constexpr int kUpscale = kFrameWidth / kWidth;

void reference_color(int x, int y, float rgb[3]) {  // lit colour of one G-buffer pixel
  int p = y * kWidth + x;
  float n[3], a[3];
  for (int c = 0; c < 3; ++c) {
    n[c] = dequantize(g_normal[c * kPixels + p], MODEL_SHADE_INPUT0_SCALE, MODEL_SHADE_INPUT0_ZERO_POINT);
    a[c] = dequantize(g_albedo[c * kPixels + p], MODEL_SHADE_INPUT1_SCALE, MODEL_SHADE_INPUT1_ZERO_POINT);
  }
  float d = dequantize(g_depth[p], MODEL_SHADE_INPUT2_SCALE, MODEL_SHADE_INPUT2_ZERO_POINT);
  float ndotl = std::max(0.0f, n[0] * kLightDir[0] + n[1] * kLightDir[1] + n[2] * kLightDir[2]);
  float light = ndotl * kKeyLight + kAmbient;
  for (int c = 0; c < 3; ++c) rgb[c] = a[c] * light * (1.0f - d) + kFogColor[c] * d;
}

void reference_filtered(int x, int y, float rgb[3]) {  // after the 3x3 Gaussian, clamped
  static const float kernel[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
  rgb[0] = rgb[1] = rgb[2] = 0.0f;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int sx = x + dx, sy = y + dy;
      if (sx < 0 || sy < 0 || sx >= kWidth || sy >= kHeight) continue;  // zero padding
      float c[3];
      reference_color(sx, sy, c);
      float k = kernel[(dy + 1) * 3 + (dx + 1)] / 16.0f;
      for (int i = 0; i < 3; ++i) rgb[i] += k * c[i];
    }
  }
  for (int i = 0; i < 3; ++i) rgb[i] = std::min(std::max(rgb[i], 0.0f), 1.0f);
}

void reference_frame_pixel(int fx, int fy, float rgb[3]) {  // after the bilinear upscale (align_corners=False)
  float sx = (fx + 0.5f) / kUpscale - 0.5f, sy = (fy + 0.5f) / kUpscale - 0.5f;
  sx = std::max(sx, 0.0f);
  sy = std::max(sy, 0.0f);
  int x0 = std::min(static_cast<int>(sx), kWidth - 1), y0 = std::min(static_cast<int>(sy), kHeight - 1);
  int x1 = std::min(x0 + 1, kWidth - 1), y1 = std::min(y0 + 1, kHeight - 1);
  float tx = sx - x0, ty = sy - y0;
  float c00[3], c01[3], c10[3], c11[3];
  reference_filtered(x0, y0, c00);
  reference_filtered(x1, y0, c01);
  reference_filtered(x0, y1, c10);
  reference_filtered(x1, y1, c11);
  for (int i = 0; i < 3; ++i) {
    rgb[i] = (1 - ty) * ((1 - tx) * c00[i] + tx * c01[i]) + ty * ((1 - tx) * c10[i] + tx * c11[i]);
  }
}

// Max abs error over a 12x20 grid of frame pixels; `frame` is RGB888 (uint8).
float check_frame(const uint8_t* frame) {
  float max_err = 0.0f;
  for (int gy = 0; gy < 20; ++gy) {
    for (int gx = 0; gx < 12; ++gx) {
      int x = gx * kFrameWidth / 12 + kFrameWidth / 24, y = gy * kFrameHeight / 20 + kFrameHeight / 40;
      float ref[3];
      reference_frame_pixel(x, y, ref);
      const uint8_t* px = frame + (static_cast<size_t>(y) * kFrameWidth + x) * 3;
      for (int c = 0; c < 3; ++c) {
        // uint8 frame byte = int8 q + 128
        float got = dequantize(static_cast<int>(px[c]) - 128, MODEL_SHADE_OUTPUT0_SCALE, MODEL_SHADE_OUTPUT0_ZERO_POINT);
        max_err = std::max(max_err, fabsf(got - ref[c]));
      }
    }
  }
  return max_err;
}

void ascii_preview(const uint8_t* frame) {
  static const char ramp[] = " .:-=+*#%@";
  constexpr int cols = 48, rows = 40;
  constexpr int cw = kFrameWidth / cols, ch = kFrameHeight / rows;
  printf("+%.*s+\n", cols, "------------------------------------------------");
  for (int r = 0; r < rows; ++r) {
    char line[cols + 1];
    for (int c = 0; c < cols; ++c) {
      float lum = 0.0f;
      for (int y = r * ch; y < (r + 1) * ch; ++y) {
        for (int x = c * cw; x < (c + 1) * cw; ++x) {
          const uint8_t* px = frame + (static_cast<size_t>(y) * kFrameWidth + x) * 3;
          lum += (0.30f * px[0] + 0.59f * px[1] + 0.11f * px[2]) / 255.0f;
        }
      }
      lum /= static_cast<float>(cw * ch);
      line[c] = ramp[std::min(9, std::max(0, static_cast<int>(lum * 9.99f)))];
    }
    line[cols] = '\0';
    printf("|%s|\n", line);
  }
  printf("+%.*s+\n", cols, "------------------------------------------------");
}

}  // namespace

// Strong override of the ExecuTorch Ethos-U backend's weak I/O copy hook
// (EthosUBackend_IoMemcpy.cpp). Two jobs: time every copy, and route the
// shade method's output, the RGB888 frame, straight into the back frame
// buffer with Helium, folding in the int8 -> uint8 offset (q + 128 is q ^
// 0x80) the display needs. The output tensor itself is left untouched.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size) {
  uint32_t t0 = cycles();
  if (size == kFrameBytes && g_frame_target != nullptr) {
#ifdef APP_HAS_DISPLAY
    // The target was on screen until the last present took effect: wait for
    // the controller to have started a frame from the other buffer.
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

  printf("NPU render: Ethos-U85 as a tensor coprocessor for a 3D pipeline, Helium on the CPU\n");
  printf("  program: %lu bytes, %d objects x %d vertices, G-buffer %dx%d int8, frame %dx%d RGB888\n",
         model_pte_size, kObjects, kMaxVertices, kWidth, kHeight, kFrameWidth, kFrameHeight);

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
  for (const auto& name : *names) {
    auto meta = module.method_meta(name);
    if (!meta.ok()) continue;
    printf("  method %s: planned memory", name.c_str());
    for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
      auto size = meta->memory_planned_buffer_size(i);
      printf(" [%u] %ld B", static_cast<unsigned>(i), size.ok() ? static_cast<long>(*size) : -1L);
    }
    printf(" (method pool %u B, temp pool %u B)\n", static_cast<unsigned>(kMethodPoolSize), static_cast<unsigned>(kTempPoolSize));
  }

  // Scene and the static vertex-stage inputs (quantized once).
  build_torus(g_meshes[0], 0, 0, 0.62f, 0.24f);
  build_sphere(g_meshes[1], kMaxVertices, g_meshes[0].triangle_count * 3, 0.30f);
  memset(g_pos_q, 0, sizeof(g_pos_q));
  memset(g_nrm_q, 0, sizeof(g_nrm_q));
  for (int o = 0; o < kObjects; ++o) {
    for (int i = 0; i < g_meshes[o].vertex_count; ++i) {
      int v = g_meshes[o].first_vertex + i;
      const Vertex& vert = g_vertices[v];
      const float p[4] = {vert.x, vert.y, vert.z, 1.0f};
      const float n[4] = {vert.nx, vert.ny, vert.nz, 0.0f};
      for (int k = 0; k < 4; ++k) {
        g_pos_q[v * 4 + k] = quantize<int16_t>(p[k], 1.0f / MODEL_VERTEX_INPUT0_SCALE, MODEL_VERTEX_INPUT0_ZERO_POINT,
                                               MODEL_VERTEX_INPUT0_QMIN, MODEL_VERTEX_INPUT0_QMAX);
        g_nrm_q[v * 4 + k] = quantize<int16_t>(n[k], 1.0f / MODEL_VERTEX_INPUT1_SCALE, MODEL_VERTEX_INPUT1_ZERO_POINT,
                                               MODEL_VERTEX_INPUT1_QMIN, MODEL_VERTEX_INPUT1_QMAX);
      }
    }
  }
  int triangles = 0;
  for (int o = 0; o < kObjects; ++o) triangles += g_meshes[o].triangle_count;
  printf("  scene: torus %d + sphere %d vertices, %d triangles\n", g_meshes[0].vertex_count, g_meshes[1].vertex_count, triangles);

  TensorBox pos_t(ScalarType::Short, kVertexShape, 3, g_pos_q);
  TensorBox nrm_t(ScalarType::Short, kVertexShape, 3, g_nrm_q);
  TensorBox mvp_t(ScalarType::Short, kMatShape, 3, g_mvp_q);
  TensorBox mv_t(ScalarType::Short, kMatShape, 3, g_mv_q);
  TensorBox normal_t(ScalarType::Char, kGBufferShape, 4, g_normal);
  TensorBox albedo_t(ScalarType::Char, kGBufferShape, 4, g_albedo);
  TensorBox depth_t(ScalarType::Char, kDepthShape, 4, g_depth);

  // Display: both frame buffers cleared to the fog colour, scan-out started.
  int back = 0;
#ifdef APP_HAS_DISPLAY
  for (int i = 0; i < 2; ++i) {
    for (size_t p = 0; p < kFrameBytes; p += 3) {
      g_framebuffer[i][p] = 26; g_framebuffer[i][p + 1] = 31; g_framebuffer[i][p + 2] = 46;
    }
    SCB_CleanDCache_by_Addr(g_framebuffer[i], static_cast<int32_t>(kFrameBytes));
  }
  int32_t dstatus = display_init();
  if (dstatus == 0) dstatus = display_start(g_framebuffer[1]);
  printf("  display: %dx%d RGB888 %s (status %ld)\n", kFrameWidth, kFrameHeight, dstatus == 0 ? "on" : "FAILED", static_cast<long>(dstatus));
  const bool display_on = dstatus == 0;
  const int frame_limit = display_on ? 0 : 8;  // 0: run forever
  const int report_every = display_on ? 120 : 1;
#else
  const bool display_on = false;
  const int frame_limit = 8;
  const int report_every = 1;
#endif

  const Mat4 projection = mat_perspective(62.0f * 3.14159265f / 180.0f,
                                          static_cast<float>(kWidth) / kHeight, 1.0f, 8.0f);
  uint32_t sum_vertex = 0, sum_vertex_copy = 0, sum_post = 0, sum_raster = 0, sum_clear = 0;
  uint32_t sum_shade = 0, sum_shade_copy = 0, sum_vsync = 0, sum_frame = 0, sum_check = 0;
  int report_frames = 0;
  float worst_error = 0.0f;
  RasterStats last_stats{0, 0, 0};
  bool pass = true;

  for (int frame = 0; frame_limit == 0 || frame < frame_limit; ++frame) {
    // The debugger's detach clears DEMCR.TRCENA and stops the cycle counter:
    // re-arm it every frame so the timings survive running standalone.
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    uint32_t t_frame = cycles();
    float t = frame * (1.0f / 50.0f);

    // 1. Matrices for this frame (CPU): the torus spins, the sphere orbits it.
    Mat4 view = mat_translate(0.0f, 0.0f, 3.4f);
    Mat4 torus_model = mat_mul(mat_rotate_x(1.1f + 0.35f * sinf(0.7f * t)), mat_rotate_y(0.9f * t));
    float orbit = 1.3f * t;
    Mat4 sphere_model = mat_mul(mat_translate(0.95f * cosf(orbit), 0.55f * sinf(orbit), 0.6f * sinf(orbit)),
                                mat_mul(mat_rotate_z(0.4f * t), mat_rotate_y(2.0f * t)));
    const Mat4 models[2] = {mat_mul(view, torus_model), mat_mul(view, sphere_model)};
    for (int o = 0; o < kObjects; ++o) {
      Mat4 mvp = mat_mul(projection, models[o]);
      quantize_matrix_transposed(mvp, g_mvp_q + o * 16, MODEL_VERTEX_INPUT2_SCALE, MODEL_VERTEX_INPUT2_ZERO_POINT,
                                 MODEL_VERTEX_INPUT2_QMIN, MODEL_VERTEX_INPUT2_QMAX);
      quantize_matrix_transposed(models[o], g_mv_q + o * 16, MODEL_VERTEX_INPUT3_SCALE, MODEL_VERTEX_INPUT3_ZERO_POINT,
                                 MODEL_VERTEX_INPUT3_QMIN, MODEL_VERTEX_INPUT3_QMAX);
    }

    // 2. Vertex stage (NPU): every object's batch in one call.
    uint32_t t0 = cycles();
    g_io_copy_cycles = 0;
    auto vertex_out = module.execute(MODEL_VERTEX_METHOD, {pos_t.evalue(), nrm_t.evalue(), mvp_t.evalue(), mv_t.evalue()});
    uint32_t t_vertex = cycles() - t0;
    uint32_t t_vertex_copy = g_io_copy_cycles;
    if (!vertex_out.ok() || vertex_out->size() != 2) {
      printf("vertex method failed (err=%u)\n", static_cast<unsigned>(vertex_out.error()));
      return 1;
    }
    const int16_t* clip = (*vertex_out)[0].toTensor().const_data_ptr<int16_t>();
    const int16_t* nview = (*vertex_out)[1].toTensor().const_data_ptr<int16_t>();

    // 3. Perspective divide and viewport (CPU, Helium).
    t0 = cycles();
    post_vertex(clip, nview, kObjects * kMaxVertices);
    uint32_t t_post = cycles() - t0;

    // 4. Rasterize into the G-buffer (CPU, Helium).
    t0 = cycles();
    clear_gbuffer();
    uint32_t t_clear = cycles() - t0;
    RasterStats stats = rasterize();
    uint32_t t_raster = cycles() - t0;

    // 5. Deferred shading, post filter, upscale (NPU); the output copy lands
    //    in the back frame buffer (see arm_ethos_io_memcpy).
    t0 = cycles();
    g_io_copy_cycles = 0;
    g_vsync_wait_cycles = 0;
    g_frame_target = g_framebuffer[back];
    auto shade_out = module.execute(MODEL_SHADE_METHOD, {normal_t.evalue(), albedo_t.evalue(), depth_t.evalue()});
    g_frame_target = nullptr;
    uint32_t t_shade = cycles() - t0;
    uint32_t t_shade_copy = g_io_copy_cycles;
    uint32_t t_vsync = g_vsync_wait_cycles;
    if (!shade_out.ok() || shade_out->empty()) {
      printf("shade method failed (err=%u)\n", static_cast<unsigned>(shade_out.error()));
      return 1;
    }

    // 6. Present: make the frame visible to the display controller's DMA and
    //    swap at the next vertical blank.
    const uint8_t* frame_rgb = g_framebuffer[back];
#ifdef APP_HAS_DISPLAY
    if (display_on) {
      SCB_CleanDCache_by_Addr(g_framebuffer[back], static_cast<int32_t>(kFrameBytes));
      display_present(g_framebuffer[back]);
      g_frame_target_free_after = display_frame_count();  // the other buffer is free once a new frame has started
      back ^= 1;
    }
#endif

    // 7. Check a grid of frame pixels against the float reference (CPU), on
    //    every reported frame.
    uint32_t t_check = 0;
    if ((frame % report_every) == report_every - 1) {
      t0 = cycles();
      float max_err = check_frame(frame_rgb);
      t_check = cycles() - t0;
      worst_error = std::max(worst_error, max_err);
      // int8 shading and a quantized bilinear resize: a few 8-bit steps of
      // error are expected, more means the NPU and the reference disagree.
      if (max_err > 8.0f / 255.0f) pass = false;
    }
    uint32_t t_total = cycles() - t_frame;

    sum_vertex += t_vertex; sum_vertex_copy += t_vertex_copy; sum_post += t_post;
    sum_raster += t_raster; sum_clear += t_clear; sum_shade += t_shade; sum_shade_copy += t_shade_copy;
    sum_vsync += t_vsync; sum_frame += t_total; sum_check += t_check;
    ++report_frames;
    last_stats = stats;

    if ((frame % report_every) == report_every - 1) {
      float n = static_cast<float>(report_frames);
      printf("frames %d..%d avg: vertex-NPU %.0f us (copy %.0f) | post-vertex %.0f us | raster %.0f us (clear %.0f; %d box px, %d covered, %d tris) | "
             "shade-NPU %.0f us (frame copy %.0f, vsync wait %.0f) | check %.0f us | frame %.1f ms = %.1f fps | max err %.1f/255\n",
             frame - report_frames + 1, frame, us(sum_vertex / n), us(sum_vertex_copy / n), us(sum_post / n),
             us(sum_raster / n), us(sum_clear / n), last_stats.pixels_tested, last_stats.pixels_covered, last_stats.triangles_drawn,
             us(sum_shade / n), us(sum_shade_copy / n), us(sum_vsync / n), us(sum_check),
             us(sum_frame / n) / 1000.0f, 1.0e6f / us(sum_frame / n), worst_error * 255.0f);
      sum_vertex = sum_vertex_copy = sum_post = sum_raster = sum_clear = sum_shade = sum_shade_copy = sum_vsync = sum_frame = sum_check = 0;
      report_frames = 0;
    }
    if (!display_on && frame == frame_limit - 1) {
      printf("last frame, luminance, 48x40 cells:\n");
      ascii_preview(frame_rgb);
    }
  }

  printf("worst NPU-vs-reference error: %.1f of 255\n", worst_error * 255.0f);
  printf("Test_result: %s\n", pass ? "PASS" : "FAIL");
  printf("\x04");  // EOT stops the FVP (uart0.shutdown_on_eot); a board just sees a 0x04
  fflush(stdout);
  return pass ? 0 : 1;
}
