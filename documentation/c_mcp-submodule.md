# c_mcp integration

`third_party/c_mcp` is the c_mcp Git submodule. Its clone URL is recorded in
`.gitmodules`. Initialize the
revision recorded by the parent repository after cloning:

```sh
git submodule update --init --recursive
```

The firmware uses the CMSIS-Pack in this checkout. The csolution declares:

```yaml
- pack: ARM::CMSIS-MCP
  path: ./third_party/c_mcp
```

Local path syntax requires a versionless pack ID and a directory containing
the PDSC; no pack installation or public index entry is required. The cproject
selects `CMSIS:MCP&Serial` instead of listing library sources.
The pack compiles `mcp.c`, `cJSON.c` and `serial_transport.c`, and copies
`Config/c_mcp_config.h` to `RTE/CMSIS/c_mcp_config.h`. The template is kept
outside the public header directory so the RTE configuration is used on board.
Renderer host tests compile the same sources and include `Config/` directly.
The library supports serial transport; a C HTTP variant is planned for a future
pack version.

Generate the archive with `bash third_party/c_mcp/gen_pack.sh`. It includes
Doxygen HTML, the simplified overview and the Python serial bridge. See the
[pack README](../third_party/c_mcp/README.md#pack-and-documentation-maintenance)
for required tools and the CMSIS dependency PDSC setting.

The pack declares a `MCP-Host` layer, `tools/SerialBridge.clayer.yml`, with
`copy-to="tools/mcp"`. Importing it from an installed pack copies the bridge,
requirements, layer file and guide into the project outside RTE. This application
already includes that copy at `tools/mcp/SerialBridge.clayer.yml`; its commands
use `tools/mcp/mcp_serial_bridge.py` for both local and installed-pack workflows.
The Python files have category `other` and are not compiled into the firmware.
The layer import is explicit and independent of the `CMSIS:MCP&Serial` component.
Refresh the copied layer explicitly when updating the pack and review its files.

Ubuntu workflows run the portable core, serial and bridge tests on each commit.
The application's host tests also run on each commit. Pack/documentation
workflows upload artifacts; gh-pages publishing remains to be configured.

`dispatch()` writes newline-delimited JSON through `mcp_serial_send()`, which
uses stdout by default. The board layer supplies strong serial RX/TX hooks.
The renderer calls the library's process_serial() loop before each frame;
`src/mcp_tools.cpp` supplies the tool handlers.
The core preserves strict envelope/trailing-input validation, notification
suppression, null error IDs, integer schemas and C++ linkage.
`CJSON_NESTING_LIMIT=16` bounds parser recursion in this project's builds.

`dispatch_with_sender()` accepts an application-provided synchronous sender.
Board defaults use CMSIS `__WEAK`; desktop builds use ordinary stdio functions.
`C_MCP_ENABLE_SERIAL_LOOP=1` enables the library input loop on the board.
All control commands use JSON-RPC. Application diagnostics remain ordinary
printf output; the library handles message framing, dispatch and sending.
This project's shared HTTP endpoint is supplied by the host Python UART
bridge.

The standalone root CMake project builds the library. Enabling
`CMCP_BUILD_TESTS` adds the sources, fixtures and CMake test definitions under
`third_party/c_mcp/tests/`. These include core tests, serial parser tests and a
stdio server fixture; Python bridge tests also live there.

The original copied implementation came from upstream commit
`b14fc8156ea6884a63b748d4f3474c6e9d101ae9`. CMSIS-MCP project files are licensed
under Apache-2.0, except the bundled `cJSON.c` and `cJSON.h`, which retain
their MIT license and original notices. cJSON comes from
[Dave Gamble's cJSON repository](https://github.com/DaveGamble/cJSON).
See [the pack license](../third_party/c_mcp/LICENSE.md) for the complete terms.
