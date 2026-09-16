// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "mcp_tools.hpp"
#include "mcp.h"
#include <cstdio>
#include <cstring>

namespace {
bool same(Color a, Color b) {
  return a.r == b.r && a.g == b.g && a.b == b.b;
}

bool call(const char* name, const char* color = nullptr, const char* tile = nullptr,
          const char* expected_status = nullptr) {
  cJSON* params = cJSON_CreateObject();
  cJSON_AddStringToObject(params, "name", name);
  cJSON* args = cJSON_AddObjectToObject(params, "arguments");
  if (color) cJSON_AddStringToObject(args, "color", color);
  if (tile) cJSON_AddStringToObject(args, "tile", tile);
  cJSON* id = cJSON_CreateNumber(1);
  cJSON* reply = handle_tools_call(id, params);
  cJSON* result = cJSON_GetObjectItemCaseSensitive(reply, "result");
  bool success = result != nullptr;
  if (success && expected_status) {
    cJSON* content = cJSON_GetObjectItemCaseSensitive(result, "content");
    cJSON* text = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(content, 0), "text");
    success = cJSON_IsString(text) && std::strstr(text->valuestring, expected_status);
  }
  cJSON_Delete(reply);
  cJSON_Delete(id);
  cJSON_Delete(params);
  return success;
}
}

int main() {
  Settings settings;
  mcp_tools_init(settings);
  struct Case { const char* input; Color rgb; const char* status; };
  const Case cases[] = {
    {"red", {1, 0, 0}, "1,0,0"}, {"blue", {0, 0, 1}, "0,0,1"},
    {"green", {0, 1, 0}, "0,1,0"}, {"yellow", {1, 1, 0}, "1,1,0"},
    {"0.25,0.5,0.75", {0.25f, 0.5f, 0.75f}, "0.25,0.5,0.75"},
  };
  struct Target { const char* tool; const char* tile; const char* label; Color* value; };
  const Target targets[] = {
    {"edgeColor", nullptr, "edge", &settings.edge},
    {"backgroundColor", nullptr, "background", &settings.background},
    {"tileColor", "a", "tile a", &settings.tile_a},
    {"tileColor", "b", "tile b", &settings.tile_b},
  };
  for (const auto& target : targets) {
    for (const auto& test : cases) {
      char status[80];
      std::snprintf(status, sizeof(status), "%s %s", target.label, test.status);
      if (!call(target.tool, test.input, target.tile) ||
          !same(*target.value, test.rgb) || !call("status", nullptr, nullptr, status) ||
          !(mcp_take_changes() & 4)) return 1;
      const Color display = color_for_display(*target.value);
#if defined(APP_DISPLAY_BGR) && APP_DISPLAY_BGR
      // First display channel is blue, last is red.
      const Color visible{display.b, display.g, display.r};
#else
      const Color visible = display;
#endif
      if (!same(visible, test.rgb)) return 2;
    }
  }
  if (!call("reset") || !same(settings.tile_a, {1, 0, 0}) ||
      !same(settings.tile_b, {0, 0, 1}) ||
      !call("status", nullptr, nullptr, "tile a 1,0,0; tile b 0,0,1")) return 3;
  free_tools();
  return 0;
}
