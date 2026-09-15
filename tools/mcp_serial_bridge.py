# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.26.0", "pyserial==3.5"]
# ///
"""Serve the board's UART tools over a shared localhost HTTP MCP endpoint.

Start once: uv run --script tools/mcp_serial_bridge.py --port COM5
Connect all MCP clients to http://127.0.0.1:8765/mcp.
Check the running server with --smoke-test (does not open the serial port).
"""

import argparse
from contextlib import asynccontextmanager
import json
import queue
import sys
import threading
import time

MAX_REQUEST_BYTES = 4095  # excludes newline; matches firmware line[4096]
MAX_RESPONSE_BYTES = 65536


class JsonLines:
    """Preserve partial serial reads and discard oversized lines in full."""

    def __init__(self):
        self.buffer = bytearray()
        self.discard = False

    def feed(self, chunk):
        for byte in chunk:
            if byte == 10:
                if not self.discard and self.buffer:
                    yield bytes(self.buffer).rstrip(b"\r")
                self.buffer.clear()
                self.discard = False
            elif not self.discard:
                if len(self.buffer) >= MAX_RESPONSE_BYTES:
                    self.buffer.clear()
                    self.discard = True
                else:
                    self.buffer.append(byte)


def parse_response(line):
    try:
        message = json.loads(line)
    except (ValueError, UnicodeError):
        return None
    if (isinstance(message, dict) and message.get("jsonrpc") == "2.0"
            and "id" in message and (("result" in message) != ("error" in message))):
        return message
    return None


def encode_request(message):
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        raise ValueError("Expected one JSON-RPC 2.0 request object")
    wire = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(wire) > MAX_REQUEST_BYTES:
        raise ValueError(f"Request exceeds the board's {MAX_REQUEST_BYTES}-byte limit")
    return wire + b"\n"


class SerialRpc:
    def __init__(self, port, timeout=15.0, diagnostics=None):
        self.port = port
        self.timeout = timeout
        self.diagnostics = diagnostics if diagnostics is not None else sys.stderr
        self.responses = queue.Queue(maxsize=16)
        self.stopped = threading.Event()
        self.reader_error = None
        # Terminate any partial request left by a previous disconnected client.
        self.port.write(b"\n")
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        lines = JsonLines()
        try:
            while not self.stopped.is_set():
                chunk = self.port.read(max(1, min(self.port.in_waiting, 4096)))
                for line in lines.feed(chunk):
                    response = parse_response(line)
                    if response is None:
                        print(line.decode("utf-8", errors="replace"), file=self.diagnostics, flush=True)
                    else:
                        self.responses.put_nowait(response)
        except Exception as exc:
            if not self.stopped.is_set():
                self.reader_error = exc

    def exchange(self, message):
        wire = encode_request(message)
        if self.reader_error is not None:
            raise RuntimeError(f"Serial reader failed: {self.reader_error}")
        if self.port.write(wire) != len(wire):
            raise RuntimeError("Incomplete UART write; request outcome is unknown")
        if "id" not in message:  # MCP notifications have no reply.
            return None
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.reader_error is not None:
                raise RuntimeError(f"Serial reader failed: {self.reader_error}")
            try:
                response = self.responses.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if response["id"] == message["id"]:
                return response
            print(f"Ignoring stale/unmatched UART reply id={response['id']!r}", file=self.diagnostics, flush=True)
        raise TimeoutError(f"No UART reply to {message['method']} within {self.timeout:g}s; request was not retried")

    def close(self):
        self.stopped.set()
        self.port.close()
        self.reader.join(timeout=1.0)


class BoardRpcError(RuntimeError):
    """A board JSON-RPC error, without a transport failure."""


class BoardGateway:
    """One UART owner with IDs and a transaction lock shared by every client.

    The lock lives in the worker thread, so cancellation of an HTTP request
    cannot release it while the serial exchange is still running.
    """

    def __init__(self, rpc):
        self.rpc = rpc
        self.lock = threading.Lock()
        self.next_id = 1
        self.failure = None

    def request(self, method, params=None, *, notification=False):
        with self.lock:
            if self.failure is not None:
                raise RuntimeError(f"UART unavailable: {self.failure}. Restart the bridge and read status before retrying a change.")
            message = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                message["params"] = params
            if not notification:
                message["id"] = self.next_id
                self.next_id += 1
            # Reject invalid/oversized requests before touching the UART. Such
            # requests must not poison an otherwise healthy shared connection.
            encode_request(message)
            try:
                response = self.rpc.exchange(message)
            except Exception as exc:
                self.failure = str(exc)
                raise RuntimeError(f"UART exchange failed: {exc}. Outcome may be unknown; no replay. Restart the bridge and read status before retrying a change.") from exc
            if response is None:
                return None
            if "error" in response:
                error = response["error"]
                raise BoardRpcError(f"Board error {error['code']}: {error['message']}")
            return response["result"]

    def close(self):
        with self.lock:
            self.failure = self.failure or "bridge stopped"
            self.rpc.close()


def open_serial_rpc(port_name, baud, timeout):
    import serial

    port = serial.Serial(port=None, baudrate=baud, timeout=0.1, write_timeout=5,
                         rtscts=False, dsrdtr=False)
    # Leave modem control lines deasserted; do not deliberately reset the board.
    port.dtr = False
    port.rts = False
    port.port = port_name
    try:
        port.open()
        return SerialRpc(port, timeout)
    except BaseException:
        port.close()
        raise


def create_http_app(port_name, baud=115200, timeout=15.0, *, rpc_factory=None):
    """Create an SDK MCP server; UART ownership follows the ASGI lifespan."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.routing import Route

    server = Server("hyperbolic-uart-bridge", version="1.0.0",
                    instructions="All clients control the same board. Changes from other clients are immediately shared.")
    state = {}

    @server.list_tools()
    async def list_tools():
        return state["tools"]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        result = await anyio.to_thread.run_sync(
            state["gateway"].request, "tools/call", {"name": name, "arguments": arguments})
        return types.CallToolResult.model_validate(result)

    manager = StreamableHTTPSessionManager(
        app=server, json_response=True, stateless=True,
        security_settings=TransportSecuritySettings(
            allowed_hosts=["127.0.0.1:*", "localhost:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*"]))

    @asynccontextmanager
    async def lifespan(app):
        factory = rpc_factory or (lambda: open_serial_rpc(port_name, baud, timeout))
        gateway = BoardGateway(await anyio.to_thread.run_sync(factory))
        state["gateway"] = gateway
        try:
            await anyio.to_thread.run_sync(gateway.request, "initialize", {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "hyperbolic-uart-bridge", "version": "1.0.0"}})
            await anyio.to_thread.run_sync(
                lambda: gateway.request("notifications/initialized", notification=True))
            result = await anyio.to_thread.run_sync(gateway.request, "tools/list")
            state["tools"] = [types.Tool.model_validate(tool) for tool in result["tools"]]
            async with manager.run():
                yield
        finally:
            # Finish any active exchange and release the port even on shutdown
            # cancellation or a failed board handshake/tool discovery.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(gateway.close)

    class McpEndpoint:
        async def __call__(self, scope, receive, send):
            await manager.handle_request(scope, receive, send)

    return Starlette(routes=[Route("/mcp", endpoint=McpEndpoint())], lifespan=lifespan)


async def smoke_test(url):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            initialized = await session.initialize()
            tools = await session.list_tools()
            status = await session.call_tool("status", {})
            print(json.dumps({"initialize": initialized.model_dump(mode="json"),
                              "tools": tools.model_dump(mode="json"),
                              "status": status.model_dump(mode="json")}, indent=2))
            if status.isError:
                raise RuntimeError("Board status failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="COM5 on Windows, /dev/ttyACM0 on Linux; required to start the server")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=15.0, help="Reply timeout in seconds; no automatic retries")
    parser.add_argument("--http-port", type=int, default=8765, help="Local HTTP port (default: 8765)")
    parser.add_argument("--smoke-test", action="store_true", help="Check the running HTTP server; does not open UART")
    args = parser.parse_args()
    if args.timeout <= 0 or args.baud <= 0:
        parser.error("timeout and baud must be positive")
    if not 1 <= args.http_port <= 65535:
        parser.error("http-port must be 1..65535")
    if not args.smoke_test and not args.port:
        parser.error("--port is required to start the server")
    try:
        import anyio
        import uvicorn

        if args.smoke_test:
            anyio.run(smoke_test, f"http://127.0.0.1:{args.http_port}/mcp")
        else:
            uvicorn.run(create_http_app(args.port, args.baud, args.timeout),
                        host="127.0.0.1", port=args.http_port, workers=1)
        return 0
    except Exception as exc:
        print(f"MCP serial bridge: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
