# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.26.0", "pyserial==3.5"]
# ///
"""Hardware-free integration tests: uv run --script tests/test_mcp_http.py -v."""

import io
import json
from pathlib import Path
import queue
import sys
import threading
import time
import unittest

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mcp_serial_bridge as bridge


class BoardSerial:
    """Delayed UART replies expose interleaved transactions and ID collisions."""

    def __init__(self):
        self.input = queue.Queue()
        self.messages = []
        self.closed = False
        self.active = 0
        self.max_active = 0
        self.drop_reply = False
        self.disconnect = False

    @property
    def in_waiting(self):
        return 1

    def read(self, count):
        if self.disconnect:
            raise OSError("USB disconnected")
        try:
            return self.input.get(timeout=0.005)
        except queue.Empty:
            return b""

    def write(self, data):
        if not data.strip():
            return len(data)
        message = json.loads(data)
        self.messages.append(message)
        if "id" not in message:
            return len(data)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.drop_reply:
            return len(data)
        response = {"jsonrpc": "2.0", "id": message["id"]}
        if message["method"] == "tools/list":
            response["result"] = {"tools": [{
                "name": "status", "description": "Firmware-provided description",
                "inputSchema": {"type": "object", "properties": {
                    "label": {"type": "string"}}}}]}
        elif message["method"] == "tools/call":
            arguments = message["params"]["arguments"]
            if message["params"]["name"] != "status" or arguments.get("label") == "invalid":
                response["error"] = {"code": -32602, "message": "Firmware rejected arguments"}
            else:
                response["result"] = {"content": [{"type": "text", "text": arguments.get("label", "board status")}]}
        else:
            response["result"] = {}

        def finish():
            self.active -= 1
            wire = json.dumps(response).encode() + b"\r\n"
            # SerialRpc must still drain logs and assemble split JSON replies.
            for chunk in (b"frame diagnostic\n", wire[:12], wire[12:]):
                self.input.put(chunk)

        timer = threading.Timer(0.02, finish)
        timer.daemon = True
        timer.start()
        return len(data)

    def close(self):
        self.closed = True


class SharedHttp(unittest.TestCase):
    def test_independent_sdk_clients_share_one_uart(self):
        async def run():
            serial = BoardSerial()
            opens = []

            def factory():
                opens.append(True)
                return bridge.SerialRpc(serial, timeout=1, diagnostics=io.StringIO())

            app = bridge.create_http_app("fake", rpc_factory=factory)
            results = {}

            async def client(label):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                    async with streamable_http_client("http://127.0.0.1:8765/mcp", http_client=http) as (reader, writer, _):
                        async with ClientSession(reader, writer) as session:
                            initialized = await session.initialize()
                            self.assertEqual(initialized.serverInfo.name, "hyperbolic-uart-bridge")
                            tools = await session.list_tools()
                            self.assertEqual(tools.tools[0].description, "Firmware-provided description")
                            result = await session.call_tool("status", {"label": label})
                            self.assertFalse(result.isError)
                            results[label] = result.content[0].text
                            error = await session.call_tool("status", {"label": "invalid"})
                            self.assertTrue(error.isError)
                            self.assertIn("-32602", error.content[0].text)
                            # A board validation error must not poison the UART.
                            self.assertFalse((await session.call_tool("status", {})).isError)

            async with app.router.lifespan_context(app):
                async with anyio.create_task_group() as group:
                    for label in ("chat-a", "chat-b", "chat-c"):
                        group.start_soon(client, label)
                # Closing all clients leaves the UART owned and a new client works.
                self.assertFalse(serial.closed)
                await client("new-chat")
            self.assertTrue(serial.closed)
            self.assertEqual(len(opens), 1)
            self.assertEqual(serial.max_active, 1)
            self.assertEqual(results, {label: label for label in ("chat-a", "chat-b", "chat-c", "new-chat")})
            ids = [m["id"] for m in serial.messages if "id" in m]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(sum(m["method"] == "initialize" for m in serial.messages), 1)
            self.assertEqual(sum(m["method"] == "tools/list" for m in serial.messages), 1)

        anyio.run(run)

    def test_host_and_origin_checks(self):
        async def run():
            serial = BoardSerial()
            app = bridge.create_http_app("fake", rpc_factory=lambda: bridge.SerialRpc(serial, timeout=1, diagnostics=io.StringIO()))
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
                    for headers in ({"host": "attacker.example"}, {"origin": "https://attacker.example"}):
                        response = await http.post("http://127.0.0.1:8765/mcp", headers=headers,
                                                   json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
                        self.assertIn(response.status_code, (403, 421))
            self.assertTrue(serial.closed)

        anyio.run(run)

    def test_failed_startup_closes_uart(self):
        async def run():
            serial = BoardSerial()
            serial.drop_reply = True
            app = bridge.create_http_app("fake", rpc_factory=lambda: bridge.SerialRpc(serial, timeout=0.03, diagnostics=io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, "no replay"):
                async with app.router.lifespan_context(app):
                    self.fail("Startup must fail when the board does not reply")
            self.assertTrue(serial.closed)
            self.assertEqual(len(serial.messages), 1)

        anyio.run(run)


class GatewayFailure(unittest.TestCase):
    def setUp(self):
        self.serial = BoardSerial()
        self.gateway = bridge.BoardGateway(bridge.SerialRpc(self.serial, timeout=0.08, diagnostics=io.StringIO()))
        self.addCleanup(self.gateway.close)

    def test_timeout_blocks_queued_requests_without_replay(self):
        self.serial.drop_reply = True
        errors = []

        def call():
            try:
                self.gateway.request("tools/call", {"name": "status", "arguments": {}})
            except RuntimeError as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=call) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 3)
        self.assertEqual(len(self.serial.messages), 1)
        self.assertTrue(any("no replay" in error for error in errors))
        self.assertEqual(sum("UART unavailable" in error for error in errors), 2)

    def test_oversized_request_does_not_poison_connection(self):
        with self.assertRaises(ValueError):
            self.gateway.request("tools/call", {"name": "status", "arguments": {"label": "x" * 4096}})
        self.assertEqual(len(self.serial.messages), 0)
        self.gateway.request("ping")
        self.assertEqual(len(self.serial.messages), 1)

    def test_disconnect_blocks_subsequent_requests(self):
        self.serial.disconnect = True
        deadline = time.monotonic() + 1
        while self.gateway.rpc.reader_error is None and time.monotonic() < deadline:
            time.sleep(0.005)
        with self.assertRaisesRegex(RuntimeError, "USB disconnected"):
            self.gateway.request("ping")
        with self.assertRaisesRegex(RuntimeError, "UART unavailable"):
            self.gateway.request("ping")
        self.assertEqual(len(self.serial.messages), 0)

    def test_cancellation_cannot_release_an_active_transaction(self):
        async def run():
            started = threading.Event()
            release = threading.Event()
            original_exchange = self.gateway.rpc.exchange

            def blocked_exchange(message):
                started.set()
                if not release.wait(timeout=2):
                    raise TimeoutError("Test did not release worker")
                return original_exchange(message)

            self.gateway.rpc.exchange = blocked_exchange
            scope = anyio.CancelScope()

            async def first():
                with scope:
                    await anyio.to_thread.run_sync(self.gateway.request, "ping")

            async with anyio.create_task_group() as group:
                group.start_soon(first)
                self.assertTrue(await anyio.to_thread.run_sync(started.wait, 1))
                scope.cancel()
                self.assertTrue(self.gateway.lock.locked())
                group.start_soon(anyio.to_thread.run_sync, self.gateway.request, "ping")
                release.set()
            self.assertEqual(len(self.serial.messages), 2)
            self.assertEqual(self.serial.max_active, 1)

        anyio.run(run)


if __name__ == "__main__":
    unittest.main()
