// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
#include "mcp_tools.hpp"
#include "mcp.h"
#include <cstdio>

namespace {
bool responded;
void capture(const char* json, int) { responded = json != nullptr; }
bool same(Color a, Color b) {
  return a.r == b.r && a.g == b.g && a.b == b.b;
}

bool call(const char* request) {
  responded = false;
  dispatch_with_sender(request, 0, capture);
  return responded;
}
}

int main() {
  Settings settings;
  mcp_tools_init(settings);
  struct Case { const char* input; Color rgb; };
  const Case cases[] = {
    {"red", {1, 0, 0}}, {"blue", {0, 0, 1}},
    {"green", {0, 1, 0}}, {"yellow", {1, 1, 0}},
    {"0.25,0.5,0.75", {0.25f, 0.5f, 0.75f}},
  };
  // These request templates use only the fixed colour literals above.
  struct Target { const char* request; Color* value; };
  const Target targets[] = {
    {R"({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"edgeColor","arguments":{"color":"%s"}}})",
     &settings.edge},
    {R"({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"backgroundColor","arguments":{"color":"%s"}}})",
     &settings.background},
    {R"({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"tileColor","arguments":{"tile":"a","color":"%s"}}})",
     &settings.tile_a},
    {R"({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"tileColor","arguments":{"tile":"b","color":"%s"}}})",
     &settings.tile_b},
  };
  for (const auto& target : targets) {
    for (const auto& test : cases) {
      char request[256];
      int length = std::snprintf(request, sizeof(request), target.request, test.input);
      if (length < 0 || static_cast<size_t>(length) >= sizeof(request)) return 1;
      if (!call(request) || !same(*target.value, test.rgb) ||
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
  if (!call(R"({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"reset","arguments":{}}})") ||
      !same(settings.tile_a, {1, 0, 0}) || !same(settings.tile_b, {0, 0, 1})) return 3;
  free_tools();
  return 0;
}
