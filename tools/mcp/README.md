# MCP host bridge layer

The `ARM::CMSIS-MCP` pack declares `SerialBridge.clayer.yml` as an
`MCP-Host` software layer. Its PDSC `copy-to="tools/mcp"` is the proposed
destination when the CMSIS Solution IDE copies the layer. Adding the pack or
selecting the `CMSIS:MCP&Serial` firmware component alone does not copy it.

To make the layer selectable in an existing project, request its type through
a variable and consume the connection provided by the pack layer:

```yaml
project:
  layers:
    - layer: $MCP-Host-Layer$
      type: MCP-Host
  connections:
    - connect: MCP application
      consumes:
        - MCP_HOST_BRIDGE
```

The layer provides `MCP_HOST_BRIDGE`. This is a matching marker for the host
bridge, not a firmware interface. Select `CMSIS:MCP&Serial` separately. This
application has already copied the host layer into `tools/mcp` and sets
`MCP-Host-Layer` for both solution targets to
`$SolutionDir()$/tools/mcp/SerialBridge.clayer.yml`.

For a new application, leave `$MCP-Host-Layer$` undefined in the csolution
until choosing the layer. Save the cproject, then run **CMSIS: Configure
Solution** from the VS Code Command Palette. Select `MCP-Host`, accept
`tools/mcp` as the copy destination or choose another directory, and click
**OK**. The IDE copies the layer, Python bridge, requirements file, and this
guide outside RTE. It also writes the selected layer path under the active
target's `variables:` in the csolution.

The PDSC `copy-to` destination is relative to the csolution directory. An
explicit `layer:` file path is relative to the cproject directory. If the
`MCP-Host` choice is disabled, check the matching `connections:` entries and
the pack path, then reload VS Code after a local pack update. The Python files
have category `other` and are not compiled into firmware.

From the csolution directory, start one bridge (adjust the port):

```sh
uv run --script tools/mcp/mcp_serial_bridge.py --port "<serial-port>"
```

Replace `<serial-port>` with your board's serial device: for example `COM3`
on Windows, `/dev/ttyACM0` or `/dev/ttyUSB0` on Linux, or
`/dev/cu.usbmodem12345` on macOS.

Python 3.10 or newer is required. `uv` manages the script's inline dependencies.
If you prefer an existing Python environment without `uv`, run these commands
with that environment's Python interpreter:

```sh
python -m pip install -r tools/mcp/requirements-mcp-serial-bridge.txt
python tools/mcp/mcp_serial_bridge.py --port "<serial-port>"
```

For an existing project Python environment or editor analysis:

```sh
uv pip install --python <venv-python> -r tools/mcp/requirements-mcp-serial-bridge.txt
```

The bridge must be the serial port's only owner. It exposes the board at
`http://127.0.0.1:8765/mcp`; Ctrl+C stops it and releases the port.
To check an already running bridge through HTTP:

```sh
uv run --script tools/mcp/mcp_serial_bridge.py --smoke-test
```

For a source checkout without the picker, copy the four files in this
directory to `tools/mcp` in the application, then set the active target's
`MCP-Host-Layer` variable to
`$SolutionDir()$/tools/mcp/SerialBridge.clayer.yml`. Keep the matching
`connections:` entry in the cproject. When updating the pack, explicitly
refresh the copied layer and review its project files;
the component's RTE configuration update does not update this host layer.
