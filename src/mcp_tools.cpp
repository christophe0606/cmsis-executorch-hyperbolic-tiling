// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "mcp_tools.hpp"
#include "mcp.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
Settings* settings;
const volatile uint32_t* frame_counter;
int changes;

int success(const char** message) { *message = "Settings updated for the next frame"; return 0; }
int invalid(const char** message, const char* error) { *message = error; return MCP_INVALID_PARAMS; }

int colour(Color& target, const char* value, const char** message) {
  Color parsed;
  if (!parse_color(value, parsed))
    return invalid(message, "Expected a known colour or r,g,b in [0,1]");
  target = parsed; changes |= 4; return success(message);
}
int edge_colour(int, const char** message, const char** args) { return colour(settings->edge, args[0], message); }
int background_colour(int, const char** message, const char** args) { return colour(settings->background, args[0], message); }
int tile_colour(int, const char** message, const char** args) {
  if (strcmp(args[0], "a") && strcmp(args[0], "b")) return invalid(message, "tile must be a or b");
  return colour(!strcmp(args[0], "a") ? settings->tile_a : settings->tile_b, args[1], message);
}
int thickness(int, const char** message, const char** args) {
  if (!parse_edge_thickness(args[0], settings->edge_thickness))
    return invalid(message, "thickness must be thin, thick or very thick");
  return success(message);
}
int antialiasing(int, const char** message, const char** args) {
  if (strcmp(args[0], "none") && strcmp(args[0], "partial") && strcmp(args[0], "full"))
    return invalid(message, "mode must be none, partial or full");
  parse_antialiasing(args[0], settings->aa);
  return success(message);
}
int animation(int, const char** message, const char** args) {
  settings->animation = !strcmp(args[0], "true"); return success(message);
}
int texture_on(int, const char** message, const char** args) {
  settings->texture = !strcmp(args[0], "true") ? TextureMode::On : TextureMode::Off;
  return success(message);
}
int video_tint(int, const char** message, const char** args) {
  settings->video_tint = !strcmp(args[0], "true"); return success(message);
}
int texture_mode(int, const char** message, const char** args) {
  if (!parse_texture_mode(args[0], settings->texture)) return invalid(message, "mode must be off, on or video");
  return success(message);
}
int geometry(int, const char** message, const char** args) {
  if (strcmp(args[0], "disk") && strcmp(args[0], "plane")) return invalid(message, "geometry must be disk or plane");
  settings->geometry = !strcmp(args[0], "plane"); changes |= 2; return success(message);
}
int symmetry(int, const char** message, const char** args) {
  double value = strtod(args[0], nullptr);
  if (value < 0 || value > 2) return invalid(message, "symmetry must be integer 0, 1 or 2");
  settings->symmetry = static_cast<int>(value); changes |= 1; return success(message);
}
int render_scale(int, const char** message, const char** args) {
  if (strcmp(args[0], "full") && strcmp(args[0], "half")) return invalid(message, "scale must be full or half");
  settings->half = !strcmp(args[0], "half"); changes |= 2; return success(message);
}
int reflections(int, const char** message, const char** args) {
  double value = strtod(args[0], nullptr);
  if (value < 1 || value > 40) return invalid(message, "iterations must be integer 1..40");
  settings->iterations = static_cast<int>(value); return success(message);
}
int texture_zoom(int, const char** message, const char** args) {
  double value = strtod(args[0], nullptr);
  if (value <= 0 || value > 100)
    return invalid(message, "zoom must be greater than 0 and at most 100");
  settings->zoom = static_cast<float>(value); settings->zoom_override = true;
  return success(message);
}
int reset_settings(int, const char** message, const char**) {
  *settings = Settings(); changes |= 7; *message = "Board defaults restored"; return 0;
}
int status(int, const char** message, const char**) {
  char* output = static_cast<char*>(mcp_arena_alloc(768));
  if (!output) { *message = "Request arena exhausted"; return MCP_INTERNAL_ERROR; }
  snprintf(output, 768,
      "scale %s, AA %s, iterations %d; symmetry %d, geometry %s, animation %s, zoom %.3g; texture %s; video tint %s; "
      "edge thickness %s; edge %.3g,%.3g,%.3g; background %.3g,%.3g,%.3g; tile a %.3g,%.3g,%.3g; tile b %.3g,%.3g,%.3g",
      settings->half ? "half" : "full", antialiasing_name(settings->aa), settings->iterations,
      settings->symmetry, settings->geometry ? "plane" : "disk", settings->animation ? "on" : "off",
      settings->zoom, texture_mode_name(settings->texture), settings->video_tint ? "on" : "off",
      edge_thickness_name(settings->edge_thickness), settings->edge.r, settings->edge.g, settings->edge.b,
      settings->background.r, settings->background.g, settings->background.b,
      settings->tile_a.r, settings->tile_a.g, settings->tile_a.b, settings->tile_b.r, settings->tile_b.g, settings->tile_b.b);
  *message = output; return 0;
}
int memory(int, const char** message, const char**) {
  char* output = static_cast<char*>(mcp_arena_alloc(192));
  if (!output) { *message = "Request arena exhausted"; return MCP_INTERNAL_ERROR; }
  snprintf(output, 192, "arena %zu/%u, high water %zu, heap allocations %zu; tree height %zu, lookup steps %zu; resource height %zu, resource steps %zu",
           mcp_arena_used(), C_MCP_ARENA_SIZE, mcp_arena_high_water(), mcp_heap_allocations(),
           mcp_tool_index_height(), mcp_tool_lookup_steps(), mcp_resource_index_height(), mcp_resource_lookup_steps());
  *message = output; return 0;
}

#if C_MCP_ENABLE_VFS
int renderer_resource(const char** message) {
  const char* settings_text = nullptr;
  int result = status(0, &settings_text, nullptr);
  if (result) { *message = settings_text; return result; }
  char* output = static_cast<char*>(mcp_arena_alloc(900));
  if (!output) { *message = "Request arena exhausted"; return MCP_INTERNAL_ERROR; }
  snprintf(output, 900, "%s; frames %lu", settings_text,
           static_cast<unsigned long>(frame_counter ? *frame_counter : 0));
  *message = output; return 0;
}
#endif

struct Definition {
  const char* name; const char* description; mcp_tool_fn callback;
  const char* argument; enum type type; const char* help;
};
const Definition definitions[] = {
  {"edgeColor", "Set edge colour", edge_colour, "color", TYPE_STR, "Colour name or r,g,b with components in [0,1]"},
  {"edgeThickness", "Set edge thickness", thickness, "thickness", TYPE_STR, "thin (original width), thick (2x), or very thick (4x)"},
  {"backgroundColor", "Set background colour", background_colour, "color", TYPE_STR, "Colour name or r,g,b with components in [0,1]"},
  {"animationOn", "Start or stop animation", animation, "on", TYPE_BOOL, "Animation enabled"},
  {"geometryType", "Select geometry", geometry, "geometry", TYPE_STR, "disk or plane"},
  {"symmetryType", "Select triangle group", symmetry, "symmetry", TYPE_INT, "0: (2,4,5), 1: (2,4,7), 2: (4,4,4)"},
  {"reset", "Reset the board renderer to its defaults", reset_settings, nullptr, TYPE_STR, nullptr},
  {"renderScale", "Select full resolution or half with Ethos upscaling", render_scale, "scale", TYPE_STR, "full or half"},
  {"antialiasing", "Select 2x2 antialiasing coverage", antialiasing, "mode", TYPE_STR, "none, partial (small triangles near the boundary), or full (whole screen)"},
  {"textureOn", "Enable texture blending or use solid tile colours", texture_on, "on", TYPE_BOOL, "true: blend texture with tile colours; false: solid tile A/B colours"},
  {"textureMode", "Select solid colours, procedural texture or live camera texture", texture_mode, "mode", TYPE_STR, "off: solid tile colours; on: procedural texture; video: live camera texture"},
  {"videoTint", "Enable or disable tile-colour tinting of live video", video_tint, "on", TYPE_BOOL, "false: original camera colours; true: blend video with tile A/B colours (default)"},
  {"reflectionLimit", "Set maximum reflection rounds", reflections, "iterations", TYPE_INT, "1 to 40"},
  {"textureZoom", "Override texture zoom", texture_zoom, "zoom", TYPE_FLOAT, "Greater than 0, at most 100"},
  {"status", "Read the current renderer settings", status, nullptr, TYPE_STR, nullptr},
  {"mcpMemory", "Read c_mcp arena and startup allocation counters", memory, nullptr, TYPE_STR, nullptr},
};
}  // namespace

void mcp_tools_init(Settings& current, const volatile uint32_t* frames) {
  settings = &current; frame_counter = frames; changes = 0; free_tools();
  for (const auto& definition : definitions) {
    auto* tool = add_tool(definition.name, definition.description);
    set_tool_callback(tool, definition.callback);
    if (definition.argument) add_argument(tool, definition.argument, definition.type, definition.help);
  }
  auto* tile = add_tool("tileColor", "Set tile A or B colour");
  set_tool_callback(tile, tile_colour);
  add_argument(tile, "tile", TYPE_STR, "a or b");
  add_argument(tile, "color", TYPE_STR, "Colour name or r,g,b in [0,1]");
#if C_MCP_ENABLE_VFS
  add_resource("hyperbolic://renderer/status", "Renderer status",
               "Current renderer settings and completed frame count", "text/plain", renderer_resource);
#endif
  mcp_prepare();
}

int mcp_take_changes() { int result = changes; changes = 0; return result; }
