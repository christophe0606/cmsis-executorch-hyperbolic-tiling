// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "mcp_tools.hpp"
#include "mcp.h"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
Settings* settings;
int changes;

bool string_is(const cJSON* item, const char* value) {
  return cJSON_IsString(item) && strcmp(item->valuestring, value) == 0;
}

bool integer_in(const cJSON* item, int low, int high) {
  return cJSON_IsNumber(item) && std::isfinite(item->valuedouble) &&
         item->valuedouble >= low && item->valuedouble <= high &&
         item->valuedouble == item->valueint;
}

void single_tool(const char* name, const char* description,
                 const char* argument, enum type type, const char* help) {
  auto* tool = add_tool(name, description);
  if (argument) add_argument(tool, argument, type, help);
}
}  // namespace

void mcp_tools_init(Settings& current) {
  settings = &current;
  changes = 0;
  free_tools();
  single_tool("edgeColor", "Set edge colour", "color", TYPE_STR,
              "Colour name or r,g,b with components in [0,1]");
  single_tool("backgroundColor", "Set background colour", "color", TYPE_STR,
              "Colour name or r,g,b with components in [0,1]");
  single_tool("animationOn", "Start or stop animation", "on", TYPE_BOOL, "Animation enabled");
  single_tool("geometryType", "Select geometry", "geometry", TYPE_STR, "disk or plane");
  single_tool("symmetryType", "Select triangle group", "symmetry", TYPE_INT,
              "0: (2,4,5), 1: (2,4,7), 2: (4,4,4)");
  single_tool("reset", "Reset the board renderer to its defaults", nullptr, TYPE_STR, nullptr);
  auto* tile = add_tool("tileColor", "Set tile A or B colour");
  add_argument(tile, "tile", TYPE_STR, "a or b");
  add_argument(tile, "color", TYPE_STR, "Colour name or r,g,b in [0,1]");
  single_tool("renderScale", "Select full resolution or half with Ethos upscaling",
              "scale", TYPE_STR, "full or half");
  single_tool("antialiasing", "Select 2x2 antialiasing coverage", "mode", TYPE_STR,
              "none, partial (small triangles near the boundary), or full (whole screen)");
  single_tool("textureOn", "Enable texture blending or use solid tile colours", "on", TYPE_BOOL,
              "true: blend texture with tile colours; false: solid tile A/B colours");
  single_tool("reflectionLimit", "Set maximum reflection rounds", "iterations", TYPE_INT, "1 to 40");
  single_tool("textureZoom", "Override texture zoom", "zoom", TYPE_FLOAT, "Greater than 0, at most 100");
  single_tool("status", "Read the current renderer settings", nullptr, TYPE_STR, nullptr);
}

int mcp_take_changes() {
  int result = changes;
  changes = 0;
  return result;
}

extern "C" cJSON* handle_tools_call(cJSON* id, cJSON* params) {
  if (!settings) return err(id, MCP_INTERNAL_ERROR, "Renderer not initialized");
  const cJSON* name = cJSON_GetObjectItemCaseSensitive(params, "name");
  const cJSON* args = cJSON_GetObjectItemCaseSensitive(params, "arguments");
  if (!cJSON_IsObject(params) || !cJSON_IsString(name) || (args && !cJSON_IsObject(args)))
    return err(id, MCP_INVALID_PARAMS, "Expected tool name and optional arguments object");

  const char* tool = name->valuestring;
  const char* message = "Settings updated for the next frame";
  const cJSON* value = nullptr;
  if (!strcmp(tool, "edgeColor") || !strcmp(tool, "backgroundColor") || !strcmp(tool, "tileColor")) {
    value = cJSON_GetObjectItemCaseSensitive(args, "color");
    Color color;
    if (!cJSON_IsString(value) || !parse_color(value->valuestring, color))
      return err(id, MCP_INVALID_PARAMS, "Expected a known colour or finite r,g,b in [0,1]");
    if (!strcmp(tool, "tileColor")) {
      const cJSON* tile = cJSON_GetObjectItemCaseSensitive(args, "tile");
      if (!string_is(tile, "a") && !string_is(tile, "b"))
        return err(id, MCP_INVALID_PARAMS, "tile must be a or b");
      (string_is(tile, "a") ? settings->tile_a : settings->tile_b) = color;
    } else {
      (!strcmp(tool, "edgeColor") ? settings->edge : settings->background) = color;
    }
    changes |= 4;
  } else if (!strcmp(tool, "antialiasing")) {
    value = cJSON_GetObjectItemCaseSensitive(args, "mode");
    const cJSON* legacy = cJSON_GetObjectItemCaseSensitive(args, "on");
    Antialiasing mode;
    if (value) {
      if (legacy || !cJSON_IsString(value) ||
          (!string_is(value, "none") && !string_is(value, "partial") && !string_is(value, "full")))
        return err(id, MCP_INVALID_PARAMS, "mode must be none, partial or full; do not also supply on");
      parse_antialiasing(value->valuestring, mode);
    } else {
      // Existing clients may still hold the old tool schema until reconnect.
      if (!cJSON_IsBool(legacy)) return err(id, MCP_INVALID_PARAMS, "mode must be none, partial or full");
      mode = cJSON_IsTrue(legacy) ? Antialiasing::Full : Antialiasing::None;
    }
    settings->aa = mode;
  } else if (!strcmp(tool, "animationOn") || !strcmp(tool, "textureOn")) {
    value = cJSON_GetObjectItemCaseSensitive(args, "on");
    if (!cJSON_IsBool(value)) return err(id, MCP_INVALID_PARAMS, "on must be boolean");
    bool& enabled = !strcmp(tool, "animationOn") ? settings->animation : settings->texture;
    enabled = cJSON_IsTrue(value);
  } else if (!strcmp(tool, "geometryType")) {
    value = cJSON_GetObjectItemCaseSensitive(args, "geometry");
    if (!string_is(value, "disk") && !string_is(value, "plane"))
      return err(id, MCP_INVALID_PARAMS, "geometry must be disk or plane");
    settings->geometry = string_is(value, "plane") ? 1 : 0;
    changes |= 2;
  } else if (!strcmp(tool, "symmetryType")) {
    value = cJSON_GetObjectItemCaseSensitive(args, "symmetry");
    if (!integer_in(value, 0, 2)) return err(id, MCP_INVALID_PARAMS, "symmetry must be integer 0, 1 or 2");
    settings->symmetry = value->valueint;
    changes |= 1;
  } else if (!strcmp(tool, "renderScale")) {
    value = cJSON_GetObjectItemCaseSensitive(args, "scale");
    if (!string_is(value, "full") && !string_is(value, "half"))
      return err(id, MCP_INVALID_PARAMS, "scale must be full or half");
    settings->half = string_is(value, "half");
    changes |= 2;
  } else if (!strcmp(tool, "reflectionLimit")) {
    value = cJSON_GetObjectItemCaseSensitive(args, "iterations");
    if (!integer_in(value, 1, 40)) return err(id, MCP_INVALID_PARAMS, "iterations must be integer 1..40");
    settings->iterations = value->valueint;
  } else if (!strcmp(tool, "textureZoom")) {
    value = cJSON_GetObjectItemCaseSensitive(args, "zoom");
    if (!cJSON_IsNumber(value) || !std::isfinite(value->valuedouble) ||
        value->valuedouble <= 0 || value->valuedouble > 100)
      return err(id, MCP_INVALID_PARAMS, "zoom must be finite, greater than 0 and at most 100");
    settings->zoom = static_cast<float>(value->valuedouble);
    settings->zoom_override = true;
  } else if (!strcmp(tool, "reset")) {
    *settings = Settings();
    changes |= 7;
    message = "Board defaults restored";
  } else if (!strcmp(tool, "status")) {
    char status[768];
    snprintf(status, sizeof(status),
             "scale %s, AA %s, iterations %d; symmetry %d, geometry %s, animation %s, zoom %.3g; texture %s; "
             "edge %.3g,%.3g,%.3g; background %.3g,%.3g,%.3g; tile a %.3g,%.3g,%.3g; tile b %.3g,%.3g,%.3g",
             settings->half ? "half" : "full", antialiasing_name(settings->aa), settings->iterations,
             settings->symmetry, settings->geometry ? "plane" : "disk", settings->animation ? "on" : "off",
             settings->zoom, settings->texture ? "on" : "off", settings->edge.r, settings->edge.g, settings->edge.b,
             settings->background.r, settings->background.g, settings->background.b,
             settings->tile_a.r, settings->tile_a.g, settings->tile_a.b,
             settings->tile_b.r, settings->tile_b.g, settings->tile_b.b);
    return ok(id, create_result_text(status));
  } else {
    return err(id, MCP_INVALID_PARAMS, "Unknown tool");
  }
  return ok(id, create_result_text(message));
}
