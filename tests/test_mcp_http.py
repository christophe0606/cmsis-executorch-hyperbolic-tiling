# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.26.0", "pyserial==3.5", "anyio", "starlette", "uvicorn"]
# ///
"""Run the bridge's portable tests from the owning c_mcp submodule."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] /
                      "third_party/c_mcp/tests/test_serial_bridge.py"), run_name="__main__")
