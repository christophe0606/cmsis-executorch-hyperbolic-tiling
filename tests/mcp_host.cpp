// Host harness uses the same dispatcher, handlers and settings as the board.
#include "mcp_tools.hpp"
#include "mcp.h"
#include "serial_transport.h"

int main() {
  Settings settings;
  mcp_tools_init(settings);
  // Mirror the board's initial apply_symmetry() before its first request.
  if (!settings.zoom_override) settings.zoom = settings.symmetry == 0 ? 1.0f : 0.5f;
  if (init_serial() != 0) return 1;
  while (process_serial() != MCP_SERIAL_EOF) {
    if ((mcp_take_changes() & 1) && !settings.zoom_override)
      settings.zoom = settings.symmetry == 0 ? 1.0f : 0.5f;
  }
  end_serial();
  free_tools();
}
