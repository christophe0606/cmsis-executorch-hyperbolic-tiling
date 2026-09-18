# c_mcp integration

`third_party/c_mcp` is a Git submodule of
[christophe0606/c_mcp](https://github.com/christophe0606/c_mcp), pinned to
`0941e3c5b59f1a021e98fb82910deee728512555`. Initialize it after cloning:

```sh
git submodule update --init --recursive
```

The firmware and host tests explicitly compile only `mcp.c` and `cJSON.c`,
using their headers from the same checkout. They do not build upstream's
HTTP transport, POSIX input loop or demo sources, or invoke its CMake project.
No local patches to the submodule are required.

This revision passes upstream's [Linux and macOS CI](https://github.com/christophe0606/c_mcp/actions/runs/35319704867),
including the portable core tests and both stdio and HTTP demo transports.

`dispatch()` writes newline-delimited JSON to retargeted stdout. The board
owns the UART input loop and `src/mcp_tools.cpp` supplies the tool handlers.
The core preserves strict envelope/trailing-input validation, notification
suppression, null error IDs, integer schemas and C++ linkage.
`CJSON_NESTING_LIMIT=16` bounds parser recursion in this project's builds.

Upstream's `dispatch_with_sender()` lets HTTP supply a response callback.
Linux/macOS builds retain both POSIX transports by default; upstream CMake
options can disable them. This project's host-side HTTP endpoint remains
provided by the Python UART bridge, independently of upstream's C HTTP code.

The original copied implementation came from upstream commit
`b14fc8156ea6884a63b748d4f3474c6e9d101ae9`. Its cJSON sources retain their MIT
notices; that upstream revision contains no separate license for the MCP
implementation. No new license is asserted.
