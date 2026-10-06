# MCP host bridge layer

The CMSIS-MCP pack provides the `MCP-Host` software layer
`SerialBridge.clayer.yml`. Import this layer from the installed pack with your
CMSIS Solution IDE. The PDSC proposes `tools/mcp` as its destination, relative
to the csolution. The layer, bridge, requirements and this guide are copied
together into the project, outside RTE.

Selecting the firmware component `CMSIS:MCP&Serial` and importing this host
layer are separate steps. The layer's Python files have category `other` and
are not compiled into the firmware.

After importing, add the copied layer to the application's cproject:

```yaml
project:
  layers:
    - layer: ./tools/mcp/SerialBridge.clayer.yml
      type: MCP-Host
```

Adjust the path if the cproject is in a subdirectory or you choose another
destination. This application's local checkout uses the same project layout.

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

For a source checkout, copy the four files in this directory to `tools/mcp`
in the application and reference the copied layer as shown above. When updating
the pack, explicitly refresh the imported layer and review its project files;
the component's RTE configuration update does not update this host layer.
