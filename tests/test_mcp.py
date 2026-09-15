"""Run: uv run --no-project tests/test_mcp.py --server tmp/mcp-tests/Debug/mcp_host.exe"""
import argparse
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mcp_serial_bridge as bridge

SERVER = None


def request(method, params=None, id=1):
    result = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        result["params"] = params
    return result


def call(name, arguments=None, id=1):
    params = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    return request("tools/call", params, id)


class FirmwareProtocol(unittest.TestCase):
    def setUp(self):
        if SERVER is None:
            self.skipTest("Pass --server after building the native C/C++ harness")

    def exchange(self, messages):
        wire = "".join((m if isinstance(m, str) else json.dumps(m)) + "\n" for m in messages)
        result = subprocess.run([SERVER], input=wire, text=True, capture_output=True, timeout=10, check=True)
        self.assertEqual(result.stderr, "")
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_initialize_discovery_notifications_and_ping(self):
        result = self.exchange([
            request("initialize", {"protocolVersion": "2025-06-18"}, "init"),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 99}},
            request("tools/list", id=2), request("ping", id=3),
        ])
        self.assertEqual([x["id"] for x in result], ["init", 2, 3])
        self.assertEqual(result[0]["result"]["protocolVersion"], "2025-06-18")
        tools = {t["name"]: t for t in result[1]["result"]["tools"]}
        self.assertEqual(len(tools), 13)
        self.assertEqual(tools["textureOn"]["inputSchema"]["properties"]["on"]["type"], "boolean")
        self.assertTrue({"edgeColor", "backgroundColor", "animationOn", "geometryType", "symmetryType", "reset"} <= tools.keys())
        self.assertEqual(tools["symmetryType"]["inputSchema"]["properties"]["symmetry"]["type"], "integer")
        self.assertEqual(result[2]["result"], {})

    def test_mutations_status_and_reset(self):
        result = self.exchange([
            call("edgeColor", {"color": "gray"}), call("backgroundColor", {"color": "0.1,0.2,0.3"}),
            call("animationOn", {"on": False}), call("geometryType", {"geometry": "plane"}),
            call("symmetryType", {"symmetry": 2}), call("renderScale", {"scale": "half"}),
            call("antialiasing", {"on": False}), call("reflectionLimit", {"iterations": 40}),
            call("tileColor", {"tile": "b", "color": "green"}), call("textureZoom", {"zoom": 2.5}),
            call("textureOn", {"on": False}),
            call("status"), call("reset"), call("status"),
        ])
        self.assertTrue(all("result" in r for r in result))
        status = result[-3]["result"]["content"][0]["text"]
        for text in ("scale half", "AA off", "iterations 40", "symmetry 2", "geometry plane",
                     "animation off", "zoom 2.5", "texture off", "edge 0.5,0.5,0.5", "tile b 0,1,0"):
            self.assertIn(text, status)
        reset = result[-1]["result"]["content"][0]["text"]
        self.assertIn("scale full, AA off, iterations 12", reset)
        self.assertIn("texture on", reset)
        self.assertIn("symmetry 0, geometry disk, animation on, zoom 1", reset)

    def test_invalid_arguments_do_not_mutate(self):
        invalid = [
            call("symmetryType", {"symmetry": 1.5}), call("symmetryType", {"symmetry": -1}),
            call("geometryType", {"geometry": "sphere"}), call("animationOn", {"on": "false"}),
            call("reflectionLimit", {"iterations": 41}), call("reflectionLimit", {"iterations": 0}),
            call("textureZoom", {"zoom": 0}), call("edgeColor", {"color": "nan,0,0"}),
            call("edgeColor", {"color": "2,0,0"}), call("edgeColor", {"color": "1,0,0junk"}),
            call("tileColor", {"tile": "c", "color": "green"}), call("renderScale", {"scale": "tiny"}),
            call("antialiasing", {}), call("does_not_exist"), call("animationOn", []),
            call("textureOn", {}), call("textureOn", {"on": "false"}), call("textureOn", {"on": 0}),
        ]
        result = self.exchange([call("status"), *invalid, call("status")])
        self.assertEqual(result[0]["result"], result[-1]["result"])
        self.assertTrue(all(r["error"]["code"] == -32602 for r in result[1:-1]))

    def test_texture_toggle_preserves_tile_colours(self):
        result = self.exchange([
            call("tileColor", {"tile": "a", "color": "yellow"}),
            call("textureOn", {"on": False}), call("status"),
            call("textureOn", {"on": True}), call("status"),
        ])
        solid = result[2]["result"]["content"][0]["text"]
        textured = result[4]["result"]["content"][0]["text"]
        self.assertIn("texture off", solid)
        self.assertIn("tile a 1,1,0; tile b 0,0,1", solid)
        self.assertEqual(solid.replace("texture off", "texture on"), textured)

    def test_bad_envelopes_and_recovery(self):
        result = self.exchange(["{broken", "[]", "{}", '{"jsonrpc":"1.0","method":"ping","id":1}',
                                '{"jsonrpc":"2.0","method":"ping","id":[]} ',
                                json.dumps(request("ping")) + " trailing",
                                request("unknown", id="missing"), request("ping", id="good")])
        self.assertEqual([r["error"]["code"] for r in result[:-1]],
                         [-32700, -32600, -32600, -32600, -32600, -32700, -32601])
        self.assertIsNone(result[0]["id"])
        self.assertEqual(result[-1], {"jsonrpc": "2.0", "id": "good", "result": {}})

    def test_depth_limit_and_repeated_calls(self):
        nested = '{"a":' * 20 + '0' + '}' * 20
        result = self.exchange([nested] + [call("status", id=i) for i in range(500)])
        self.assertEqual(result[0]["error"]["code"], -32700)
        self.assertEqual(len(result), 501)
        self.assertEqual(result[-1]["id"], 499)


class FakeSerial:
    def __init__(self):
        self.input = queue.Queue()
        self.writes = []
        self.closed = False

    @property
    def in_waiting(self):
        return 1

    def read(self, count):
        try:
            return self.input.get(timeout=0.01)
        except queue.Empty:
            return b""

    def write(self, data):
        self.writes.append(data)
        if data.strip():
            message = json.loads(data)
            if "id" in message:
                for chunk in [b"frame 123: diagnostic\r\n", b'{"jsonrpc":"2.0",',
                              json.dumps({"id": message["id"], "result": {}})[1:].encode() + b"\r\n"]:
                    self.input.put(chunk)
        return len(data)

    def close(self):
        self.closed = True


class HostBridge(unittest.TestCase):
    def test_partial_noise_crlf_and_oversize_recovery(self):
        lines = bridge.JsonLines()
        self.assertEqual(list(lines.feed(b'{"jsonrpc":')), [])
        frames = list(lines.feed(b'"2.0","id":1,"result":{}}\r\nframe log\n'))
        self.assertEqual(bridge.parse_response(frames[0])["id"], 1)
        self.assertIsNone(bridge.parse_response(frames[1]))
        self.assertEqual(list(lines.feed(b"x" * (bridge.MAX_RESPONSE_BYTES + 1) + b"\nOK\n")), [b"OK"])

    def test_request_limit_uses_compact_bytes(self):
        with self.assertRaises(ValueError):
            bridge.encode_request(request("ping", {"big": "x" * 4096}))
        self.assertTrue(bridge.encode_request(request("ping")).endswith(b"\n"))

    def test_serial_end_to_end(self):
        serial = FakeSerial()
        diagnostics = io.StringIO()
        rpc = bridge.SerialRpc(serial, timeout=1, diagnostics=diagnostics)
        try:
            self.assertIsNone(rpc.exchange({"jsonrpc": "2.0", "method": "notifications/initialized"}))
            self.assertEqual(rpc.exchange(request("ping", id="abc")),
                             {"jsonrpc": "2.0", "id": "abc", "result": {}})
            self.assertIn("frame 123", diagnostics.getvalue())
            self.assertEqual(serial.writes[0], b"\n")
        finally:
            rpc.close()
        self.assertTrue(serial.closed)

    def test_timeout_does_not_replay(self):
        serial = FakeSerial()
        serial.write = lambda data: serial.writes.append(data) or len(data)
        rpc = bridge.SerialRpc(serial, timeout=0.03, diagnostics=io.StringIO())
        try:
            with self.assertRaises(TimeoutError):
                rpc.exchange(call("reset"))
            self.assertEqual(len(serial.writes), 2)  # initial newline + one request
        finally:
            rpc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server")
    args, remaining = parser.parse_known_args()
    SERVER = str(Path(args.server).resolve()) if args.server else None
    unittest.main(argv=[sys.argv[0], *remaining])
