# Current state

## User instructions

- Do not commit c_mcp until the user explicitly authorizes committing again.
- Test implementation changes on the board; host testing alone is insufficient.
- Preserve upstream cJSON.c and cJSON.h without modifications.

## Serial transport cleanup — complete, uncommitted (2026-10-05)

- Removed third_party/c_mcp/c_mcp_compiler.h and its compiler/linker fallbacks.
- Renamed stdio_transport.c/.h to serial_transport.c/.h. Stdio is the default
  implementation of the serial API.
- Renamed the hooks to mcp_serial_transport_init(), mcp_serial_transport_close(),
  mcp_serial_getchar() and mcp_serial_send(); renamed the optional input loop to
  init_serial(), process_serial() and end_serial().
- Updated serial configuration names, CMake, CI, demo, board integration,
  Doxygen examples and documentation. The new names replace the former names.
- Desktop definitions are ordinary stdio functions. Board definitions use CMSIS
  __WEAK from cmsis_compiler.h; application overrides remain strong.
- Adapted the existing parser regression test to mock stdio, without desktop
  weak-symbol overrides.

## Validation

- Windows host: core/parser CTest 3/3, default stdio tests 3/3, renderer CTest
  6/6, protocol tests 14/14, bridge tests 11/11: 37 tests passed.
- Doxygen generated the public MCP and serial API documentation with zero warnings.
- CMSIS DevKit-E8@Release build b-30 succeeded. Programming task d-31 exited 0;
  the debugger subsequently reported running with a responsive DAP.
- On-board acceptance through the existing Python HTTP MCP bridge passed:
  17 tools discovered, 18 callback mutations checked, 100 repeated status
  requests, 25 resource reads, invalid-input and arena-exhaustion recovery.
  Renderer frames advanced from 155 to 338; heap allocation count remained 480.
  Evidence: tmp/serial-transport-board.json.
- The board exercises its strong serial RX/TX hooks and existing line parser.
  The library's optional serial input loop is tested on the host.
- Upstream cJSON SHA-256 hashes remain unchanged:
  cJSON.c: D3ED55CE8FFF1023AA90325E64B829CF0C174481030CBD62794074E3E455AE0D
  cJSON.h: 451C02BF3F52534CA933E61B7745F376F376E40883CEF09321086DB06D56204D

## Library contents cleanup - complete, uncommitted (2026-10-05)

- Removed the old root demo main.c, tools.c/.h, processing.c and config.h.
  Their useful stdio regression coverage now lives in tests/serial_driver.c
  and tests/test_serial_transport.py.
- Removed http.c/.h, CMCP_BUILD_HTTP and the unused legacy handle_fetch()
  metadata helper. The C library supports serial transport; the separate
  host Python UART-to-HTTP bridge remains available.
- Removed the obsolete mixed transport test, debug input file and stale
  machine-specific demo launch configuration.
- Moved all test target definitions into third_party/c_mcp/tests/CMakeLists.txt.
  The root CMake file builds the library and optionally includes tests.
  CTest now also runs the four default stdio Python tests.
- Updated documentation and CI. CI now covers serial library/tests on Linux,
  macOS and Windows; the changed CI jobs have not been run remotely.
- Fresh Windows library-only build passed with the default serial loop enabled.
  Test build with CMCP_BUILD_SERIAL=OFF passed; no Python dependency is required
  for a library-only build.
- Host validation: c_mcp CTest 4/4 (three C executables plus a four-test Python
  suite), renderer CTest 6/6, protocol 14/14 and bridge 11/11.
  Total: 38 individual host tests passed. Doxygen had zero warnings.
- CMSIS DevKit-E8@Release build b-32 succeeded; programming task d-33 exited 0.
  The debugger then reported running with a responsive DAP.
- Board acceptance passed: 17 tools, 18 callback mutation checks, 100 repeated
  status requests, 25 resource reads, invalid-input and arena-exhaustion
  recovery. Frames advanced from 301 to 485; heap allocation count stayed 480.
  Evidence: tmp/serial-only-board.json. Green edges and original settings restored.
- cJSON.c/.h hashes remain unchanged. No commits were made.

## Application serial API upgrade - complete, uncommitted (2026-10-05)

- app_main.cpp now initializes through init_serial(), calls process_serial()
  once before each frame, applies mcp_take_changes(), and calls end_serial()
  when its finite run completes. C_MCP_ENABLE_SERIAL_LOOP is enabled.
- Removed poll_console(), its duplicate line buffer, and the legacy text command
  parser (including preview/probe helpers). All incoming control commands are
  JSON-RPC. Existing application printf diagnostics remain ordinary output.
- No text/line callback extension was retained. The application has no UART RX/TX,
  framing, dispatch or message-send implementation.
- Moved strong initialization/RX/TX overrides into the DevKit-E8 board layer.
  The Corstone board adapter supplies initialization and nonblocking empty RX.
- Updated the renderer host fixture to use the same serial processing API and
  added rejection/non-mutation coverage for the removed text commands.
- Updated current controls and integration documentation to describe JSON MCP.
- Initial board launch b-34/d-35 exited during startup. Live CMSIS debugging
  established that default transport initialization called setvbuf(), which
  returned 1 on this Arm C library; app_main consequently returned failure.
  Board initialization now reuses the idempotent stdio_init(). Temporary
  breakpoints were removed. No diagnostic printf instrumentation was added.
- Corrected Release build b-37 succeeded; programming task d-38 exited 0.
  The debugger then reported running with a responsive DAP.
- Board acceptance through the existing bridge passed: 17 tools, 18 callback
  mutations, 100 repeated status requests, 25 resource reads, invalid-input
  and arena-exhaustion recovery. Frames advanced 19 -> 202; heap count stayed 480.
  Evidence: tmp/process-serial-board.json.
- Additional UART testing used only CMSIS serial tools with the bridge released.
  Four legacy text commands returned MCP parse errors without changing settings.
  Partial JSON plus CRLF, embedded-null discard/recovery, and an oversized
  4200-byte line followed by a valid request all passed. Frames advanced
  578 -> 1127. Evidence: tmp/process-serial-framing.json.
- Host suites passed: three core/parser tests, four default stdio tests, six
  renderer tests, fifteen protocol tests and eleven bridge tests (39 total).
  Doxygen generated without warnings; cJSON.c/.h hashes remain unchanged.
- Restarted the bridge after closing CMSIS serial ownership: PID 17124,
  exec session 14730, COM5, http://127.0.0.1:8765/mcp.
  Original renderer settings, including green edges, were restored and confirmed
  through MCP. The board remains running with debugger attached. No commits.

## Running state

The board runs the tested Release firmware with debugger attached. The existing
Python bridge remains running on COM5 at http://127.0.0.1:8765/mcp. Original
renderer settings were restored, including green edges. No commits were made
for this cleanup; all changes remain available for review.
