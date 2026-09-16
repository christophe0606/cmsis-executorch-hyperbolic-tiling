// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once

struct Color {
  float r, g, b;
};
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

struct Settings {
  int symmetry = 0;        // 0: (2,4,5), 1: (2,4,7), 2: (4,4,4) triangle group
  int geometry = 0;        // 0: disk, 1: plane (strip model)
  bool half = false;
  Antialiasing aa = Antialiasing::Partial;
  bool texture = true;    // false: solid tile A/B colours
  int iterations = 12;
  bool animation = true;   // Moebius drift
  float zoom = 1.0f;       // texture zoom (the presets set it, "zoom" overrides)
  bool zoom_override = false;
  Color edge{0.0f, 0.0f, 0.0f};
  Color background{0.0f, 0.0f, 0.0f};
  Color tile_a{1.0f, 0.0f, 0.0f};
  Color tile_b{0.0f, 0.0f, 1.0f};
  EdgeThickness edge_thickness = EdgeThickness::Thin;
};

bool parse_color(const char* name, Color& out);
