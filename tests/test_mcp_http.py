# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.26.0", "pyserial==3.5", "anyio", "starlette", "uvicorn"]
# ///
"""Run portable bridge tests against the application's copied MCP host layer."""
from pathlib import Path
import os
import runpy

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CMCP_BRIDGE_DIR", str(root / "tools/mcp"))
    runpy.run_path(str(root / "third_party/c_mcp/tests/test_serial_bridge.py"), run_name="__main__")
