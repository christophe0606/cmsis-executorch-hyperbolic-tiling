// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstring>
#include <cmath>
#include "tiling_settings.hpp"

bool parse_texture_mode(const char* name, TextureMode& out) {
  if (!strcmp(name, "off")) out = TextureMode::Off;
  else if (!strcmp(name, "on")) out = TextureMode::On;
  else if (!strcmp(name, "video")) out = TextureMode::Video;
  else return false;
  return true;
}

bool parse_edge_thickness(const char* name, EdgeThickness& out) {
  if (!strcmp(name, "thin")) out = EdgeThickness::Thin;
  else if (!strcmp(name, "thick")) out = EdgeThickness::Thick;
  else if (!strcmp(name, "very thick")) out = EdgeThickness::VeryThick;
  else return false;
  return true;
}

bool parse_antialiasing(const char* name, Antialiasing& out) {
  if (!strcmp(name, "none") || !strcmp(name, "off")) out = Antialiasing::None;
  else if (!strcmp(name, "partial")) out = Antialiasing::Partial;
  else if (!strcmp(name, "full") || !strcmp(name, "on")) out = Antialiasing::Full;
  else return false;
  return true;
}

struct NamedColor {
  const char* name;
  Color color;
};
const NamedColor kColors[] = {
    {"black", {0, 0, 0}},     {"white", {1, 1, 1}},     {"red", {1, 0, 0}},        {"green", {0, 1, 0}},
    {"blue", {0, 0, 1}},      {"yellow", {1, 1, 0}},    {"cyan", {0, 1, 1}},       {"magenta", {1, 0, 1}},
    {"orange", {1, 0.5f, 0}}, {"grey", {0.5f, 0.5f, 0.5f}}, {"gray", {0.5f, 0.5f, 0.5f}}, {"navy", {0.05f, 0.08f, 0.2f}},
};

bool parse_color(const char* name, Color& out) {
  for (const NamedColor& c : kColors) {
    if (strcmp(name, c.name) == 0) {
      out = c.color;
      return true;
    }
  }
  float r, g, b;
  char extra;
  if (sscanf(name, "%f,%f,%f%c", &r, &g, &b, &extra) == 3 &&
      std::isfinite(r) && std::isfinite(g) && std::isfinite(b) &&
      r >= 0 && r <= 1 && g >= 0 && g <= 1 && b >= 0 && b <= 1) {
    out = {r, g, b};
    return true;
  }
  return false;
}

