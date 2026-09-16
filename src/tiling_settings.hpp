// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

struct Color {
  float r, g, b;
};

// Settings and MCP always use RGB. Convert each colour triple for rendering;
// the DevKit-E8 display maps its first channel to blue.
inline Color color_for_display(Color rgb) {
#if defined(APP_DISPLAY_BGR) && APP_DISPLAY_BGR
  return {rgb.b, rgb.g, rgb.r};
#else
  return rgb;
#endif
}
enum class Antialiasing { None, Partial, Full };
inline const char* antialiasing_name(Antialiasing mode) {
  return mode == Antialiasing::Full ? "full" : mode == Antialiasing::Partial ? "partial" : "none";
}
bool parse_antialiasing(const char* name, Antialiasing& out);

enum class EdgeThickness { Thin, Thick, VeryThick };
inline const char* edge_thickness_name(EdgeThickness thickness) {
  return thickness == EdgeThickness::VeryThick ? "very thick" :
         thickness == EdgeThickness::Thick ? "thick" : "thin";
}
inline float edge_width(EdgeThickness thickness) {
  // Hyperbolic distance; thin preserves the original renderer width.
  return thickness == EdgeThickness::VeryThick ? 0.04f :
         thickness == EdgeThickness::Thick ? 0.02f : 0.01f;
}
bool parse_edge_thickness(const char* name, EdgeThickness& out);

enum class TextureMode : uint8_t { Off, On, Video };
inline const char* texture_mode_name(TextureMode mode) {
  return mode == TextureMode::Video ? "video" : mode == TextureMode::On ? "on" : "off";
}
bool parse_texture_mode(const char* name, TextureMode& out);

struct Settings {
  int symmetry = 1;        // 0: (2,4,5), 1: (2,4,7), 2: (4,4,4) triangle group
  int geometry = 0;        // 0: disk, 1: plane (strip model)
  bool half = false;
  Antialiasing aa = Antialiasing::None;
  TextureMode texture = TextureMode::Video;
  bool video_tint = true;  // Blend live video with tile A/B colours.
  int iterations = 12;
  bool animation = true;   // Moebius drift
  float zoom = 1.0f;       // texture zoom (the presets set it, "zoom" overrides)
  bool zoom_override = false;
  Color edge{1.0f, 1.0f, 1.0f};
  Color background{0.0f, 0.0f, 0.0f};
  Color tile_a{1.0f, 0.0f, 0.0f};
  Color tile_b{0.0f, 0.0f, 1.0f};
  EdgeThickness edge_thickness = EdgeThickness::Thin;
};

bool parse_color(const char* name, Color& out);
