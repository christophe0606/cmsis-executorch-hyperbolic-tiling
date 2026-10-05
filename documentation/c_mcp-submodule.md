# c_mcp integration

`third_party/c_mcp` is a Git submodule of
[christophe0606/c_mcp](https://github.com/christophe0606/c_mcp). Initialize the
revision recorded by the parent repository after cloning:

```sh
git submodule update --init --recursive
```

The firmware and renderer host tests explicitly compile `mcp.c`, `cJSON.c`
and `serial_transport.c`, using headers from the same checkout. The library
supports serial transport and has no C HTTP transport or application demo.

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
`b14fc8156ea6884a63b748d4f3474c6e9d101ae9`. Its cJSON sources retain their MIT
notices; that upstream revision contains no separate license for the MCP
implementation. No new license is asserted.
