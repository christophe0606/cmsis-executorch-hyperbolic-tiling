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
#include "tiling_kernel.hpp"
#ifdef APP_DUAL_CORE
#include "tiling_worker.hpp"
#endif
#include "mcp_tools.hpp"
#include "mcp.h"
#ifdef APP_HAS_CAMERA
#include "board_camera.h"
#endif

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

// Debugger-visible controls and measurements; do not affect visual settings.
volatile uint32_t g_tiling_use_he = 1;
struct FrameMetrics {
  uint32_t frames, render_cycles, geometry_cycles, upscale_cycles;
  uint32_t he_tiles, tile_count, he_wait_cycles;
  uint32_t worker_ready, worker_clock, validation_errors, validation_cases;
};
volatile FrameMetrics g_tiling_metrics{};

namespace {

#ifdef APP_DUAL_CORE
tiling::WorkerClient g_worker;
#endif

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
#ifdef APP_TEMP_POOL_SECTION
#define APP_TEMP_POOL_ATTRIBUTES __attribute__((section(APP_TEMP_POOL_SECTION)))
#else
#define APP_TEMP_POOL_ATTRIBUTES APP_POOL_ATTRIBUTES
#endif
#ifdef APP_FRAMEBUFFER_SECTION
#define APP_FRAMEBUFFER_ATTRIBUTES __attribute__((section(APP_FRAMEBUFFER_SECTION)))
#else
#define APP_FRAMEBUFFER_ATTRIBUTES APP_POOL_ATTRIBUTES
#endif

constexpr size_t kMethodPoolSize = APP_METHOD_POOL_SIZE;
constexpr size_t kTempPoolSize = APP_TEMP_POOL_SIZE;
alignas(16) uint8_t g_method_pool[kMethodPoolSize] APP_POOL_ATTRIBUTES;
alignas(16) uint8_t g_temp_pool[kTempPoolSize] APP_TEMP_POOL_ATTRIBUTES;

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
tiling::RenderState g_render;
auto& g_texture = g_render.texture;
auto& g_colors = g_render.colors;
// Separable disk/strip mapping tables for all possible subpixel coordinates.
auto& g_map_x = g_render.map_x;
auto& g_map_y = g_render.map_y;
auto& g_map_sin = g_render.map_sin;

// Two RGB888 frame buffers the display controller scans out.
alignas(32) uint8_t g_framebuffer[2][kFrameBytes] APP_FRAMEBUFFER_ATTRIBUTES;

// ---------------------------------------------------------------------------
// Settings: what the original demo's MCP tools change.
// ---------------------------------------------------------------------------
auto& g_settings = g_render.settings;

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

auto& g_planes = g_render.planes;

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


void build_maps() {
  int width = g_settings.half ? kFrameWidth / 2 : kFrameWidth;
  int height = g_settings.half ? kFrameHeight / 2 : kFrameHeight;
  g_render.aa_center = tiling::partial_center(g_settings, width, height);
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
    const Color display = color_for_display(*c[i]);
    g_colors[i][0] = q_texture(display.r);
    g_colors[i][1] = q_texture(display.g);
    g_colors[i][2] = q_texture(display.b);
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
#ifdef APP_DUAL_CORE
bool validate_worker() {
  // Exercise transport + identical kernels on different physical cores. These
  // private settings never change the displayed scene or the MCP settings.
  static tiling::RenderState test;
  alignas(32) static uint16_t remote[3 * kPixels];
  test = g_render;
  for (int mode = 0; mode < 48; ++mode) {
    test.settings.half = mode & 1;
    const int shade = (mode >> 1) & 3;
    test.settings.texture = shade == 0 ? TextureMode::Off : shade == 1 ? TextureMode::On : TextureMode::Video;
    test.settings.video_tint = shade != 3;
    test.settings.geometry = (mode >> 3) & 1;
    test.settings.aa = static_cast<Antialiasing>(mode / 16);
    test.settings.iterations = 40;
    const int width = test.settings.half ? kFrameWidth / 2 : kFrameWidth;
    const int height = test.settings.half ? kFrameHeight / 2 : kFrameHeight;
    test.aa_center = tiling::partial_center(test.settings, width, height);
    const float offsets[] = {0, -0.25f, 0.25f};
    for (int sample = 0; sample < 3; ++sample) {
      for (int x = 0; x < width; ++x) {
        auto map = tiling::map_horizontal(x, width, offsets[sample], test.settings.geometry);
        test.map_x[sample][x] = map.value;
        test.map_sin[sample][x] = map.sine;
      }
      for (int y = 0; y < height; ++y)
        test.map_y[sample][y] = tiling::map_vertical(y, width, height, offsets[sample], test.settings.geometry);
    }
    g_worker.begin_frame(test, 1.25f);
    // Center, boundary transition, and clamped top/bottom halo strips.
    for (int by : {height / 2, test.aa_center.y0 - 1, -1, height - 2}) {
      const int bx = width - kWidth;
      g_worker.submit(bx, by);
      auto expected = tiling::render_strip(test, g_accum, 1.25f, bx, by);
      tiling::GeometryStats actual{};
      if (!g_worker.wait(remote, actual)) {
        ++g_tiling_metrics.validation_errors;
        return false;
      }
      ++g_tiling_metrics.validation_cases;
      if (memcmp(remote, g_accum, sizeof(remote)) != 0 ||
          actual.rounds != expected.rounds || actual.vectors != expected.vectors ||
          actual.capped_vectors != expected.capped_vectors) {
        ++g_tiling_metrics.validation_errors;
        return false;
      }
      if (shade == 3) {
        // Untinted video must be independent of both tile colours, including
        // antialiased edges and the full/half-resolution halo boundaries.
        for (int tile = 0; tile < 2; ++tile)
          for (int c = 0; c < 3; ++c) test.colors[tile][c] = -1 - test.colors[tile][c];
        tiling::render_strip(test, g_accum, 1.25f, bx, by);
        for (int tile = 0; tile < 2; ++tile)
          for (int c = 0; c < 3; ++c) test.colors[tile][c] = -1 - test.colors[tile][c];
        ++g_tiling_metrics.validation_cases;
        if (memcmp(remote, g_accum, sizeof(remote)) != 0) {
          ++g_tiling_metrics.validation_errors;
          return false;
        }
      }
    }
  }
  return true;
}
#endif

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
  char* arg = strtok(nullptr, strcmp(cmd, "edge-thickness") == 0 ? "\r\n" : " \t");
  if (arg && strcmp(cmd, "edge-thickness") == 0) {
    while (*arg == ' ' || *arg == '\t') ++arg;
    size_t length = strlen(arg);
    while (length && (arg[length - 1] == ' ' || arg[length - 1] == '\t')) arg[--length] = '\0';
  }
  if (strcmp(cmd, "help") == 0) {
    printf("commands: scale full|half | aa none|partial|full | iterations 1..40 | symmetry 0|1|2 | geometry disk|plane | animation on|off | edge <color> | background <color> | "
           "edge-thickness thin|thick|very thick | tile a|b <color> | texture on|off|video | video-tint on|off | zoom <f> | reset | status | preview | probe x y  (colours: names or r,g,b)\n");
  } else if (strcmp(cmd, "edge-thickness") == 0) {
    if (!arg || !parse_edge_thickness(arg, g_settings.edge_thickness)) {
      printf("use thin|thick|very thick\n");
      return 0;
    }
    printf("edge thickness %s\n", edge_thickness_name(g_settings.edge_thickness));
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
    if (!parse_antialiasing(arg, g_settings.aa)) { printf("use none|partial|full\n"); return 0; }
    printf("AA %s\n", antialiasing_name(g_settings.aa));
  } else if (strcmp(cmd, "texture") == 0 && arg) {
    if (!parse_texture_mode(arg, g_settings.texture)) { printf("use on|off|video\n"); return 0; }
    printf("texture %s\n", arg);
  } else if (strcmp(cmd, "video-tint") == 0 && arg) {
    if (strcmp(arg, "on") && strcmp(arg, "off")) { printf("use on|off\n"); return 0; }
    g_settings.video_tint = strcmp(arg, "on") == 0;
    printf("video tint %s\n", arg);
  } else if (strcmp(cmd, "iterations") == 0 && arg) {
    char* end;
    long n = strtol(arg, &end, 10);
    if (*end || n < 1 || n > 40) { printf("iterations must be 1..40\n"); return 0; }
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
    printf("scale %s, AA %s, iterations %d\n", g_settings.half ? "half" : "full", antialiasing_name(g_settings.aa), g_settings.iterations);
    printf("symmetry %d, geometry %s, animation %s, zoom %.2f\n", g_settings.symmetry,
           g_settings.geometry ? "plane" : "disk", g_settings.animation ? "on" : "off", g_settings.zoom);
    printf("texture %s\n", texture_mode_name(g_settings.texture));
    printf("video tint %s\n", g_settings.video_tint ? "on" : "off");
    printf("edge thickness %s\n", edge_thickness_name(g_settings.edge_thickness));
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
  update_texture(0);
#ifdef APP_DUAL_CORE
  g_tiling_metrics.worker_ready = g_worker.init();
  if (g_worker.available()) g_tiling_metrics.worker_clock = g_worker.worker_clock();
  if (g_worker.available() && !validate_worker()) g_tiling_use_he = 0;
  printf("HE renderer: %s (%lu MHz)\n", g_worker.available() ? "ready" : "HP fallback",
         static_cast<unsigned long>(g_tiling_metrics.worker_clock / 1000000));
  printf("HE validation: %lu strips, %lu errors\n",
         static_cast<unsigned long>(g_tiling_metrics.validation_cases),
         static_cast<unsigned long>(g_tiling_metrics.validation_errors));
#endif
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
#ifdef APP_FRAME_PERF_LOG
  uint64_t cycles_since_report = 0;
#endif
  for (int frame = 0; display_on || frame < 2; ++frame) {
    DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    int changed = poll_console();
    if (changed & 1) apply_symmetry();
    if (changed & 3) build_maps();
    if (changed & 4) quantize_colors();
    uint32_t texture_start = cycles();
    if (g_settings.texture == TextureMode::Video) {
#ifdef APP_HAS_CAMERA
      // The snapshot is updated before either core starts rendering this frame.
      int32_t camera_result = camera_update_texture(g_texture, kTextureSize);
#else
      int32_t camera_result = -1;
#endif
      if (camera_result < 0) {
        printf("video texture unavailable (%ld); restoring texture on\n", static_cast<long>(camera_result));
        g_settings.texture = TextureMode::On;
      }
    }
    if (g_settings.texture != TextureMode::Video) {
#ifdef APP_HAS_CAMERA
      camera_stop();
#endif
      if (g_settings.texture == TextureMode::On) update_texture(g_settings.animation ? animation_time : 0);
    }
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
    uint64_t geometry_cycles = 0, upscale_cycles = 0;
#ifdef APP_FRAME_PERF_LOG
    uint32_t rounds = 0, vectors = 0, capped = 0;
#endif
    const int columns = width / kWidth;
    const int tile_count = columns * ((height + step - 1) / step);
    int next_tile = 0, he_tile = -1;
    uint32_t he_tiles = 0, he_wait_cycles = 0;
    const uint32_t render_start = cycles();
#ifdef APP_DUAL_CORE
    const bool use_he = g_tiling_use_he && g_worker.available();
    if (use_he) {
      g_worker.begin_frame(g_render, animation_time);
      he_tile = next_tile++;
      g_worker.submit(0, -halo);
    }
#endif
    while (next_tile < tile_count || he_tile >= 0) {
      int tile;
      bool remote = false;
#ifdef APP_DUAL_CORE
      remote = he_tile >= 0 && (next_tile == tile_count || g_worker.ready());
#endif
      tile = remote ? he_tile : next_tile++;
      const int bx = (tile % columns) * kWidth;
      const int by = (tile / columns) * step;
      tiling::GeometryStats stats{};
      bool received = false;
      const uint32_t geometry_start = cycles();
#ifdef APP_DUAL_CORE
      if (remote) {
        const uint32_t wait_start = cycles();
        received = g_worker.wait(g_accum, stats);
        he_wait_cycles += cycles() - wait_start;
        he_tile = -1;
        if (received) ++he_tiles;
        // Release the result buffer and dispatch the next strip before HP
        // performs conversion/Ethos work on the copied result in local DTCM.
        if (g_worker.available() && next_tile < tile_count) {
          he_tile = next_tile++;
          g_worker.submit((he_tile % columns) * kWidth,
                          (he_tile / columns) * step - halo);
        }
      }
#endif
      if (!received) stats = tiling::render_strip(g_render, g_accum, animation_time, bx, by - halo);
      geometry_cycles += cycles() - geometry_start;
#ifdef APP_FRAME_PERF_LOG
      rounds += stats.rounds; vectors += stats.vectors; capped += stats.capped_vectors;
#endif
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
    }
    total_cycles += static_cast<uint32_t>(cycles() - render_start);
    g_tiling_metrics.render_cycles = static_cast<uint32_t>(total_cycles);
    g_tiling_metrics.geometry_cycles = static_cast<uint32_t>(geometry_cycles);
    g_tiling_metrics.upscale_cycles = static_cast<uint32_t>(upscale_cycles);
    g_tiling_metrics.he_tiles = he_tiles;
    g_tiling_metrics.tile_count = tile_count;
    g_tiling_metrics.he_wait_cycles = he_wait_cycles;
    g_tiling_metrics.frames = frame + 1;
    g_last_frame = g_framebuffer[back];
#ifdef APP_HAS_DISPLAY
    if (display_on) {
      SCB_CleanDCache_by_Addr(g_framebuffer[back], static_cast<int32_t>(kFrameBytes));
      if (display_present(g_framebuffer[back]) != 0) { printf("present failed\n"); return 1; }
      g_frame_target_free_after = display_frame_count();
      back ^= 1;
    }
#endif
#ifdef APP_FRAME_PERF_LOG
    // UART performance reports are enabled only in Debug and Benchmark builds.
    cycles_since_report += total_cycles;
    if (frame == 0 || cycles_since_report >= SystemCoreClock) {
      printf("frame %d: %s AA %s iter %d | geometry %.1f ms, upscale %.1f ms | render %.1f ms | rounds/vector %.2f, capped %lu/%lu\n",
           frame, g_settings.half ? "half" : "full", antialiasing_name(g_settings.aa), g_settings.iterations,
           us(geometry_cycles) / 1000, us(upscale_cycles) / 1000, us(total_cycles) / 1000,
           static_cast<float>(rounds) / vectors, static_cast<unsigned long>(capped), static_cast<unsigned long>(vectors));
      printf("  HE %lu/%d strips, transfer/wait %.2f ms\n",
             static_cast<unsigned long>(he_tiles), tile_count, us(he_wait_cycles) / 1000);
      cycles_since_report = 0;
    }
#endif
    if (g_settings.animation) animation_time += us(total_cycles) * 1.0e-6f;
  }
  printf("Test_result: PASS\n\x04");
  return 0;
}
