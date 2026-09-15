// Host harness uses the same dispatcher, handlers and settings as the board.
#include "mcp_tools.hpp"
#include "mcp.h"
#include <iostream>
#include <string>

int main() {
  Settings settings;
  mcp_tools_init(settings);
  std::string line;
  while (std::getline(std::cin, line)) {
    dispatch(line.c_str(), 0);
    if ((mcp_take_changes() & 1) && !settings.zoom_override)
      settings.zoom = settings.symmetry == 0 ? 1.0f : 0.5f;
  }
  free_tools();
}
