# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial==3.5"]
# ///
"""Forward MCP stdio to the board's newline-delimited JSON-RPC over UART.

Run with uv run tools/mcp_serial_bridge.py --port COM5. Only JSON-RPC replies
go to stdout. Console diagnostics go to stderr. Requests are serialized; a
timeout ends the connection without replaying an operation with unknown outcome.
"""

import argparse
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


def serve(rpc, source, sink):
    for line in source:
        if not line.strip():
            continue
        message = json.loads(line)
        try:
            response = rpc.exchange(message)
        except ValueError as exc:
            if not isinstance(message, dict) or "id" not in message:
                raise
            response = {"jsonrpc": "2.0", "id": message["id"],
                        "error": {"code": -32600, "message": str(exc)}}
        if response is not None:
            print(json.dumps(response, separators=(",", ":"), allow_nan=False), file=sink, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="COM5 on Windows, /dev/ttyACM0 on Linux")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=15.0, help="Reply timeout in seconds; no automatic retries")
    parser.add_argument("--smoke-test", action="store_true", help="Initialize, list tools and read status, then exit")
    args = parser.parse_args()
    if args.timeout <= 0 or args.baud <= 0:
        parser.error("timeout and baud must be positive")
    import serial

    rpc = None
    try:
        # Leave modem control lines deasserted; do not deliberately reset the board.
        port = serial.Serial(port=None, baudrate=args.baud, timeout=0.1, write_timeout=5,
                             rtscts=False, dsrdtr=False)
        port.dtr = False
        port.rts = False
        port.port = args.port
        port.open()
        rpc = SerialRpc(port, args.timeout)
        if args.smoke_test:
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                    "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "serial-smoke-test", "version": "1.0"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "status"}},
            ]
            for message in requests:
                response = rpc.exchange(message)
                if response is not None:
                    print(json.dumps(response), flush=True)
                    if "error" in response:
                        raise RuntimeError("Board returned a JSON-RPC error")
        else:
            serve(rpc, sys.stdin, sys.stdout)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, serial.SerialException) as exc:
        print(f"MCP serial bridge: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if rpc is not None:
            rpc.close()


if __name__ == "__main__":
    raise SystemExit(main())
