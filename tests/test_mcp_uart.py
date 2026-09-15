# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial==3.5"]
# ///
"""Board regression: uv run --script tests/test_mcp_uart.py --port COM5.

Changes renderer settings for the test, then restores board defaults. The
serial bridge must not already be running. Uses only the board's MCP server.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import serial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mcp_serial_bridge import SerialRpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--output", default="out/mcp-uart-validation.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    port = serial.Serial(port=None, baudrate=115200, timeout=0.1, write_timeout=5)
    port.dtr = False
    port.rts = False
    port.port = args.port
    port.open()
    with output.with_suffix(".log").open("w", encoding="utf-8") as diagnostics:
        rpc = SerialRpc(port, diagnostics=diagnostics)

        def exchange(method, params=None, error=None):
            message = {"jsonrpc": "2.0", "id": len(records) + 1, "method": method}
            if params is not None:
                message["params"] = params
            start = time.monotonic()
            response = rpc.exchange(message)
            records.append({"method": method, "params": params, "response": response,
                            "latency_ms": round((time.monotonic() - start) * 1000, 2)})
            if error is None:
                assert "result" in response, response
            else:
                assert response["error"]["code"] == error, response
            return response

        def call(name, arguments=None, error=None):
            params = {"name": name}
            if arguments is not None:
                params["arguments"] = arguments
            return exchange("tools/call", params, error)

        try:
            exchange("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "uart-regression", "version": "1.0"}})
            rpc.exchange({"jsonrpc": "2.0", "method": "notifications/initialized"})
            tools = exchange("tools/list")["result"]["tools"]
            assert len(tools) == 13
            call("reset")
            call("antialiasing", {"on": True})
            call("reflectionLimit", {"iterations": 40})
            # A large request arrives during full-resolution AA rendering.
            # Write in short chunks to exercise reception across UART IRQs.
            original_write = port.write
            def fragmented_write(data):
                return sum(original_write(data[i:i + 17]) for i in range(0, len(data), 17))
            port.write = fragmented_write
            exchange("ping", {"padding": "x" * 3500})
            port.write = original_write
            call("renderScale", {"scale": "half"})
            call("antialiasing", {"on": False})
            call("reflectionLimit", {"iterations": 12})
            call("animationOn", {"on": False})
            call("geometryType", {"geometry": "plane"})
            call("symmetryType", {"symmetry": 2})
            call("edgeColor", {"color": "gray"})
            call("backgroundColor", {"color": "0.1,0.2,0.3"})
            call("tileColor", {"tile": "b", "color": "green"})
            call("textureZoom", {"zoom": 2.5})
            call("textureOn", {"on": False})
            before = call("status")["result"]
            text = before["content"][0]["text"]
            for expected in ("scale half", "AA off", "animation off", "geometry plane",
                             "symmetry 2", "zoom 2.5", "texture off", "edge 0.5,0.5,0.5", "tile b 0,1,0"):
                assert expected in text, text
            for name, arguments in (
                ("symmetryType", {"symmetry": 1.5}), ("reflectionLimit", {"iterations": 41}),
                ("edgeColor", {"color": "nan,0,0"}), ("edgeColor", {"color": "2,0,0"}),
                ("tileColor", {"tile": "c", "color": "red"}), ("textureZoom", {"zoom": -1}),
                ("animationOn", {"on": "false"}), ("geometryType", {"geometry": "sphere"}),
                ("textureOn", {"on": "false"}),
            ):
                call(name, arguments, error=-32602)
            assert call("status")["result"] == before
            call("textureOn", {"on": True})
            assert "texture on" in call("status")["result"]["content"][0]["text"]
            for _ in range(20):
                exchange("ping")
            call("reset")
            final = call("status")["result"]["content"][0]["text"]
            assert "scale full, AA off, iterations 12" in final, final
            assert "texture on" in final, final
            print(f"PASS: {len(records)} board requests; all tools, invalid input, 3.5KB fragmented request, repeated calls; defaults restored")
        finally:
            rpc.close()
            output.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
