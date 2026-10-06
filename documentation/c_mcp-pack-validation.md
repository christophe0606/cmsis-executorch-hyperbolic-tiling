# c_mcp CMSIS-Pack validation

Validated locally on 2026-10-05. No commits, pushes or documentation deployment
were performed.

## Pack and documentation

- Archive: `third_party/c_mcp/output/ARM.CMSIS-MCP.0.1.0.pack`.
- PackChk 1.4.5: schema, static dependencies and RTE model checks passed with
  zero errors and zero warnings.
- Doxygen 1.13.2: generation passed; local links in all 33 HTML pages resolved.
- ZIP integrity and every PDSC component file reference passed.
- Packed sources and configuration match the checkout used for the board build.

## Host tests

Run locally on Windows with MSVC 19.38 and Python 3.13:

- c_mcp CTest: 4/4 passed with the serial loop enabled; 4/4 passed with it disabled.
  Both runs include the VFS-disabled core, serial parser and stdio fixture.
- Application CTest: 6/6 passed.
- Application protocol/bridge unit tests: 15/15 passed.
- Hardware-free HTTP serial bridge integration tests: 11/11 passed.
- CMSIS csolution and cproject YAML schema validation passed.

The new GitHub workflows select Ubuntu and Python 3.12. They have not been run
on GitHub because the changes remain uncommitted. Automatic tests use the host;
the previous firmware/FVP workflow is now manual.

## Board

The CMSIS tools built and loaded the `DevKit-E8@Release` target with AC6 6.24,
including both M55_HP and M55_HE images. The generated application build resolved
`ARM::CMSIS-MCP@0.1.0` from `third_party/c_mcp` and selected
`ARM::CMSIS:MCP&Serial@0.1.0`. The component uses the application's
`RTE/CMSIS/c_mcp_config.h` and `CJSON_NESTING_LIMIT=16`.

JSON-RPC requests through the CMSIS serial tools on COM5 at 115200 baud verified
initialization, `status`, `mcpMemory`, ping and the renderer status resource.
Memory diagnostics reported a 32768-byte arena, a 2611-byte high water mark and
480 startup heap allocations. Renderer resource reads reported frame counts
457, 649 and 2054, confirming continued rendering. The final read succeeded after
the debug session was closed, leaving the application running and the probe free.

After adopting the ARM::CMSIS-MCP name, pack generation, PackChk, documentation
link checks and the E8 Release build/load were repeated successfully. Ping and
renderer resource reads succeeded on the renamed pack build (frame count 121).

## Host layer validation (2026-10-06)

The pack now declares `tools/SerialBridge.clayer.yml` as an `MCP-Host` layer
with `copy-to="tools/mcp"`. The application references its project-local copy
at `tools/mcp/SerialBridge.clayer.yml`.

- Regenerated pack: PackChk passed with zero errors and zero warnings.
- Layer YAML schema and all declared file references passed.
- Copied the four host files from the archive into a fresh test project's
  `tools/mcp` using the PDSC's declared paths; they match the application's copy.
- Bridge integration suite against that unpacked copy: 11/11 passed.
- Application protocol/bridge tests using the project-local copy: 15/15 passed.
- Doxygen generation and local links in all 33 HTML pages passed.
- E8 Release build with the host layer passed. Its Python and requirements
  files retain category `other` in the generated build description.

The existing CMSIS Run task remained active. This host-layer change required
no board reload. Layer import through an IDE wizard was not exercised; the
PDSC declaration, copy destination, copied content and project integration
were validated locally.

## Bash automation documentation (2026-10-06)

The library README and Doxygen automation page contain the same short,
commented Bash/curl example. It initializes an MCP HTTP session, calls the
example `setGain` tool with the numeric argument `gain: 2.0`, without discovery,
and closes the session.

- Extracted the example from both documents and checked they match.
- Bash syntax and the complete script passed against the real Python bridge
  with a simulated UART fixture advertising `setGain`. The simulated board
  received exactly `{"name":"setGain","arguments":{"gain":2.0}}`, session
  deletion succeeded and the shared bridge remained active.
- Doxygen generation and local links in all 35 HTML pages passed.
- Regenerated pack: PackChk passed with zero errors and zero warnings.

This documentation change required no firmware build or board reload.

## API documentation wording (2026-10-06)

Removed AVL, balancing, traversal-complexity and linked-list explanations
from the library README and public API comments. The usage documentation
now describes exact-name/URI matching, callback invocation, missing entries
and handle lifetimes. Optional lookup metrics remain in the diagnostics group.

- No AVL references remain in the API documentation or generated HTML.
- Doxygen generation and local links in all 35 HTML pages passed.
- Regenerated pack: PackChk passed with zero errors and zero warnings.
- Archive integrity and packaged source/document consistency passed.

Only documentation and header comments changed; no runtime behavior changed.

## Boolean alias removal (2026-10-06)

Removed `set_boolean_argument_alias()` from the public API, implementation
and documentation. Removed the application's antialiasing alias registration.
Antialiasing uses its declared string argument `mode` with `none`, `partial`
or `full`; the old boolean `on` argument is rejected. Declared `TYPE_BOOL`
arguments continue to work. Updated host and board acceptance tests accordingly.

- Core CTest: 4/4 passed with the serial loop enabled and 4/4 with it disabled,
  including the resources-disabled build.
- Application CTest: 6/6 passed.
- Application protocol/bridge suite: 15/15 passed, including rejection of
  `on: true` and `on: false` without modifying settings.
- E8 Release build and load/run succeeded through the CMSIS tools. The CMSIS
  Run task remains active.
- Runtime verification passed through the CMSIS serial tools on COM6 at
  115200 baud. The board advertises only the string `mode` argument for
  antialiasing. Both old boolean inputs and mixed `mode`/`on` input returned
  `MCP_INVALID_PARAMS` (-32602), with settings unchanged. Ping and renderer
  resource reads passed; frame counts advanced from 1759 to 1945.
- The verification connection was closed after testing, leaving the port
  available to the bridge.
- Doxygen generation and links in all 35 HTML pages passed. PackChk reported
  zero errors and zero warnings; archive integrity and source consistency passed.
- The removed API is absent from the generated pack's sources and documentation.
