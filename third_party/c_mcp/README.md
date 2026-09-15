# Embedded c_mcp port

`mcp.c` and `mcp.h` originate from christophe0606/c_mcp, commit
`b14fc8156ea6884a63b748d4f3474c6e9d101ae9`, as checked out in the original
`shader_linux_glsl/c_mcp` submodule. The original source has no license header
or standalone license file in that checkout; no new license is asserted for it.
`cJSON.c` and `cJSON.h` are copied unchanged from the same checkout and retain
their MIT license notices.

Local changes remove POSIX/HTTP dependencies, send newline-delimited JSON to
stdio (UART4 on the board), validate request envelopes and trailing input,
suppress notification replies, emit null error IDs, and use integer schemas.
The application supplies the tool handlers in `src/mcp_tools.cpp`.
`CJSON_NESTING_LIMIT=16` bounds parser recursion in the firmware build.
