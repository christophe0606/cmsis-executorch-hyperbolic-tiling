// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstring>
#include <cmath>
#include "tiling_settings.hpp"

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

