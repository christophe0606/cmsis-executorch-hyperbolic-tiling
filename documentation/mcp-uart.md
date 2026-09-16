# MCP over UART4 (no RTOS)

The board runs the original Linux demo's `c_mcp` dispatcher, adapted to bare
metal. One persistent host bridge owns the UART and exposes a localhost
Streamable HTTP MCP endpoint shared by all Codex chats:

```text
Codex chats <-> http://127.0.0.1:8765/mcp <-> one UART bridge
                                                    |
                                         COM5, 115200 8N1
                                                    |
                                           UART4 IRQ buffer
                                                    |
                                         MCP call between frames
```

The original six tools retain their names and arguments: `edgeColor(color)`,
`backgroundColor(color)`, `animationOn(on)`, `geometryType(geometry)`,
`symmetryType(symmetry)`, and `reset()`. The board also exposes
`tileColor(tile,color)`, `renderScale(scale)`, `antialiasing(mode)`,
`reflectionLimit(iterations)`, `textureZoom(zoom)`, `textureOn(on)`, `textureMode(mode)`,
`videoTint(on)`, `edgeThickness(thickness)`, and `status()`.
Run `tools/list` to see their schemas. Colours include both `gray` and `grey`,
or comma-separated RGB values in [0,1]. Reset uses the board's defaults:
full resolution, AA partial, 12 reflection rounds, animated disk, symmetry 0,
texture on, thin edges, and red/blue tile colours.

All colour names, numeric inputs and `status()` values use standard RGB:
`red` is `1,0,0`, `green` is `0,1,0`, and `yellow` is `1,1,0`. Clients must not
swap red and blue. The DevKit-E8 board layer sets `APP_DISPLAY_BGR=1`; firmware
converts each edge, background and tile RGB triple to the display channel
order when preparing renderer colours, for both full and half resolution.
Settings, reset defaults and the UART console retain RGB values. Camera and
procedural texture pixels are unchanged by this colour conversion.

Use `edgeThickness(thickness="thin")`, `edgeThickness(thickness="thick")`, or
`edgeThickness(thickness="very thick")` to choose edge width. These use
hyperbolic distances 0.01 (the original width), 0.02, and 0.04 respectively,
in both geometries and at either render resolution. Changes apply to the next
frame. `status()` reports the thickness; `reset()` restores thin edges.
The UART console equivalent is `edge-thickness thin|thick|very thick`.

The C firmware remains responsible for tool definitions, argument validation,
execution and UART JSON-RPC. The Python `mcp` package handles the HTTP MCP
transport and client handshakes. At startup the bridge initializes its one
board connection and discovers the tool schemas; it does not duplicate them
in Python. HTTP tool calls are forwarded to the board. After the COM connection
closes or fails, the bridge automatically reopens it and repeats initialization
and complete tool discovery. It publishes the new list only after discovery
succeeds. Changes to tool names, descriptions or schemas trigger the standard
MCP `notifications/tools/list_changed` notification for connected clients;
unchanged definitions (including a different list order) do not. The bridge
advertises `capabilities.tools.listChanged: true` and uses stateful Streamable
HTTP sessions with a GET/SSE channel for server notifications. Clients should
refresh `tools/list` when notified. A firmware update that leaves the COM
connection healthy does not trigger rediscovery; restart the bridge in that case.

Plane geometry uses a strip rotated 90 degrees clockwise to fill the portrait
display. Use `geometryType(geometry="plane")` to select it.
Use `textureOn(on=false)` for solid tile colours, then `tileColor(tile="a",
color="red")` and `tileColor(tile="b",color="blue")` to choose the two colours.
Edges, animation and antialiasing still work. `textureOn(on=true)` restores
the texture blend without changing the selected colours. The equivalent
UART console command is `texture on|off|video`; `status()` reports the current mode.

`textureMode(mode="video")` uses the DevKit-E8 MT9M114 MIPI camera as the texture.
`mode="on"` selects the procedural texture and `mode="off"` selects solid colours.
The legacy `textureOn(on=true)` always selects procedural texture, including when
switching from video. Camera updates continue with animation off; zoom, tile colours,
geometry, antialiasing and both render scales apply to video as usual.

Use `videoTint(on=false)` to disable tile-colour tinting and show the camera's
original colours. `videoTint(on=true)` restores the default 50/50 blend with tile
colours. The console equivalent is `video-tint off|on`. This setting affects only
video, preserves tile A/B colours, and is remembered across texture-mode changes.
Edges, background, antialiasing and upscaling still apply. `status()` reports
`video tint on|off`; `reset()` restores tinting on.

The camera captures 320x320 RGB565 snapshots into a 200 KiB buffer in SRAM0, then
resamples each completed snapshot to the renderer's 128x128 planar int8 texture.
The next capture overlaps rendering, while both cores use an immutable texture.
Until the first camera frame arrives, the previous texture remains visible. Leaving
video mode stops capture. A camera error or a two-second capture timeout restores
procedural mode with a console diagnostic; selecting video again retries camera
setup unless stopping capture failed, which requires a board reset. Targets without
a camera also fall back to procedural texture. The board layer uses the camera
initialization sequence from ModelNova's vStream VideoIn and the installed pack's
MT9M114, CPI, CSI2, and I2C drivers; no RTOS or vStream wrapper is needed.

## Why no RTOS is needed

Rendering, parsing and applying settings all run in the main loop. The CMSIS
USART callback receives bytes into an 8 KiB ring during rendering and replies.
At the start of each frame, the foreground consumes at most one complete line,
dispatches it, and updates the geometry/maps/colours before drawing. There are
no renderer/MCP threads, locks around settings, or JSON work in an interrupt.
An accepted change affects the next frame; its reply acknowledges the settings
update, not completion of the LCD scanout.

Polling only once per frame would leave the hardware FIFO unattended for
hundreds of milliseconds in full-resolution AA mode. Interrupt reception
avoids that dependency. Tool latency still includes the remainder of the
current frame and transmitting the response at 115200 baud. An RTOS would
be useful only if control calls must complete independently of frame duration.

UART4 uses `RTE_UART4_BLOCKING_MODE_ENABLE=0`. The local `retarget_stdio.c`
owns the receive callback and CMSIS-Compiler character hooks; the pack's
separate stdin/stdout/stderr retarget components are removed from this board
layer. Transmit waits for driver completion in the foreground, with interrupts
enabled. Terminal commands still work, on the same port. The FVP retains its
existing output-only console path; UART MCP is enabled on DevKit-E8.

## Build and connect

1. Build `cmsis-executorch.Debug+DevKit-E8` and load/run the image through
   the CMSIS Developer Assistant in VS Code. Do not invoke pyOCD or GDB
   directly or install another copy of pyOCD.
2. Set SW4 to **UART4** and use **PRG USB**, **115200 8N1**, with no hardware
   flow control. COM5 was the detected USB serial port on this Windows host;
   verify it if cables/devices change.
3. Close Serial Monitor and any other process holding that port. The bridge
   must be its only owner, including when using CMSIS Assistant serial tools.
4. Start the shared server **once**, from the workspace, and leave it running:

   ```powershell
   uv run --script tools/mcp_serial_bridge.py --port COM5
   ```

   `uv` installs `mcp==1.26.0` and `pyserial==3.5` in its script environment
   on first use. The server binds only to `127.0.0.1`, port `8765`, and opens
   one COM5 handle at a time, reopening after connection failures. Ctrl+C stops
   it and releases the UART. Startup still requires a responding board.
   Do not run multiple server instances or use multiple ASGI workers/reload.
   Use `--http-port` to change the HTTP port, and update the Codex URL to match.
5. From another terminal, test the running server:

   ```powershell
   uv run --script tools/mcp_serial_bridge.py --smoke-test
   ```

   This connects through HTTP, initializes an MCP client, lists the tools and
   reads status. It neither opens the UART nor changes renderer settings.

## Workspace configuration for Codex

Create **`.codex/config.toml` in the workspace root**, and copy the table from
[`codex-mcp.toml`](codex-mcp.toml) into it. Merge it with any existing settings.
Install `uv` on your PATH and adjust `COM5` in the server startup command to
your board's serial port. Codex connects to the running server by URL; it
does not start a bridge per chat. The server must be running before Codex
connects. The configuration contains no checkout-specific absolute paths.

```toml
[mcp_servers.hyperbolic_uart]
url = "http://127.0.0.1:8765/mcp"
startup_timeout_sec = 30
tool_timeout_sec = 60
default_tools_approval_mode = "approve"
```

The server-wide approval setting automatically approves all tools exported by
the board, including future additions. Per-tool approval entries are unnecessary.

Codex supports stdio and Streamable HTTP MCP transports; it does not open a
serial port directly. Its project configuration is loaded for trusted projects.
See the [official MCP configuration documentation](https://developers.openai.com/codex/mcp/).
When migrating from the stdio configuration, disconnect the old Codex MCP
connection/process first so it releases COM5, then start the HTTP server.
Reload/reconnect Codex after changing the configuration; existing connections
can retain the old settings. `codex mcp list` can verify the configuration
from the workspace. The example file is supplied for other checkouts to copy;
this workspace's `.codex/config.toml` already selects the HTTP endpoint.

All chats see and change the same renderer state. Closing a chat does not
close the UART. Complete UART requests/replies are serialized across clients,
and the bridge assigns its own unique request IDs so IDs reused by different
chats cannot collide. Serialization is per tool call: a sequence of calls
from one chat can be interleaved with calls from another chat.

## Transport limits and recovery

Requests and replies are single-line JSON-RPC, terminated by LF (CRLF also
works). Input lines are limited to 4095 bytes and JSON nesting to 16 levels.
The bridge compacts requests and rejects oversized requests before writing.
It serializes requests rather than pipelining them, preserving ring capacity
while a slow frame completes. Notifications have no reply. Ordinary board
logs are continuously drained to bridge stderr. HTTP client lifecycle messages
are handled by the SDK, independently of the single board connection. Board
JSON-RPC tool errors are returned to HTTP clients as MCP tool error results
with the board's error code and message.

Firmware discards damaged/overflowed input through the next newline rather
than executing a truncated command. The bridge also bounds response lines.
On timeout, disconnect or incomplete write the bridge marks the UART
unavailable and rejects queued and subsequent calls without writing to it.
It closes the failed handle before trying to reopen the same configured port,
checking once per second while idle. Reconnection attempts continue while the
port is absent, busy, or the board handshake/discovery fails. During recovery,
the last successfully discovered list remains available but tool calls fail.
After a successful handshake and discovery, new calls use the new connection;
queued calls on the failed connection are rejected and never replayed.
Correct the connection/run state in VS Code if needed, allow the bridge to
reconnect, and read `status` before repeating a change. Cancelling an HTTP request does
not abort or replay an in-progress UART transaction; a submitted change may
still complete. Each reopened connection sends an initial newline
to terminate any partial input left by a disconnected client. It does not
deliberately toggle DTR/RTS to reset the board.

## Host tests

The host harness compiles the same dispatcher, tool handlers and settings as
the firmware, without the renderer or board driver. From a CMake-capable shell:

```powershell
cmake -S tests -B tmp/mcp-tests
cmake --build tmp/mcp-tests --config Debug
ctest --test-dir tmp/mcp-tests -C Debug --output-on-failure
uv run --no-project tests/test_mcp.py --server tmp/mcp-tests/Debug/mcp_host.exe -v
uv run --script tests/test_mcp_http.py -v
```

For a single-configuration generator or Linux, use `tmp/mcp-tests/mcp_host`
as the executable path. Omit `--server` to run only the Python bridge tests.
Tests cover the handshake, tool settings, invalid input, notifications,
parser recovery, repeated calls, partial serial reads, diagnostic filtering,
request size limits and no replay on timeout. The HTTP tests use independent
Python MCP SDK clients and a simulated serial device to check shared ownership,
concurrent requests, ID isolation, client reconnection, cancellation, error
recovery, startup cleanup and localhost Host/Origin checks. Real localhost HTTP
and SSE tests also cover automatic COM reconnection, failed rediscovery,
paginated tool lists, notification delivery to multiple clients, unchanged lists,
schema changes, tool removal and no replay after timeout. Hardware validation
is required separately for IRQ reception and display coexistence.

## Validated on this board (2026-09-15)

The hardware results below predate the shared HTTP adapter. The HTTP adapter
passed seven hardware-free gateway/SDK integration tests, and the four serial
bridge tests still pass. The new HTTP path has not yet been validated on the
physical board; the six optional native firmware tests were not run for this
host-only change.

- AC6 6.24.0 DevKit-E8 build passed. Parser and argument validation use
  `-ffp-mode=full` so non-finite input checks work independently of renderer
  optimization. Existing RTE update notices remain.
- Loaded with the CMSIS VS Code extension's bundled pyOCD 0.45.1 through
  ULINKplus `L85986697A`; no pyOCD installation was performed.
- COM5 smoke test passed: initialization, 12-tool discovery and status while
  full-resolution AA rendering continued (observed frame render time ~325 ms).
- `uv run --script tests/test_mcp_uart.py --port COM5` passed 47 requests:
  every tool, invalid-input rejection without changes, a 3.5 KiB request sent
  in short chunks during full-AA/40-round rendering, and 20 repeated pings.
  Defaults were restored and the port closed. Local evidence is in
  `out/mcp-uart-validation.json` and `.log` (ignored build outputs).
- Four Python bridge tests passed. The optional native C/C++ test harness
  was configured but not built/run because its build approval was declined.

These checks establish UART MCP operation alongside rendering during the
test; they are not a long-duration stability test of the renderer.
