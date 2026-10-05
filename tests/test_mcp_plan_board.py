# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp==1.26.0", "anyio"]
# ///
"""Acceptance tests through the existing HTTP MCP bridge; never opens UART."""
import argparse
import json
from pathlib import Path
import re
import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError


def text(result):
    assert not result.isError, result
    return result.content[0].text


async def run(task, url, output, no_vfs):
    async with streamable_http_client(url) as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            initialized = await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"status", "mcpMemory", "tileColor", "textureMode"} <= names, names
            baseline = text(await session.call_tool("status", {}))
            resource_first = resource_last = None
            if task >= 4 and no_vfs:
                assert initialized.capabilities.resources is None
                assert (await session.list_resources()).resources == []
                try:
                    await session.read_resource("hyperbolic://renderer/status")
                except McpError as exc:
                    assert exc.error.code == -32601
                else:
                    raise AssertionError("Disabled resources must reject reads")
            if task >= 4 and not no_vfs:
                assert initialized.capabilities.resources
                resources = await session.list_resources()
                assert any(str(resource.uri) == "hyperbolic://renderer/status" for resource in resources.resources)
                resource_first = (await session.read_resource("hyperbolic://renderer/status")).contents[0].text
                assert resource_first.startswith(baseline + "; frames ")
            memory_before = text(await session.call_tool("mcpMemory", {}))
            heaps = re.search(r"heap allocations (\d+)", memory_before).group(1)
            original_edge = re.search(r"; edge ([0-9][^;]+)", baseline).group(1)
            callback_checks = 0
            try:
                text(await session.call_tool("edgeColor", {"color": "green"}))
                assert "; edge 0,1,0;" in text(await session.call_tool("status", {}))
                if task >= 4 and not no_vfs:
                    assert "; edge 0,1,0;" in (await session.read_resource("hyperbolic://renderer/status")).contents[0].text
                text(await session.call_tool("edgeColor", {"color": original_edge}))
                invalid = await session.call_tool("edgeColor", {"color": "nan,0,0"})
                assert invalid.isError
                assert text(await session.call_tool("status", {})) == baseline
                if task >= 3:
                    cases = [
                        ("backgroundColor", {"color": "0.1,0.2,0.3"}, "background 0.1,0.2,0.3"),
                        ("tileColor", {"color": "yellow", "tile": "a"}, "tile a 1,1,0"),
                        ("edgeThickness", {"thickness": "very thick"}, "edge thickness very thick"),
                        ("animationOn", {"on": False}, "animation off"),
                        ("geometryType", {"geometry": "plane"}, "geometry plane"),
                        ("geometryType", {"geometry": "disk"}, "geometry disk"),
                        ("symmetryType", {"symmetry": 2}, "symmetry 2"),
                        ("renderScale", {"scale": "full"}, "scale full"),
                        ("renderScale", {"scale": "half"}, "scale half"),
                        ("antialiasing", {"mode": "partial"}, "AA partial"),
                        ("antialiasing", {"on": False}, "AA none"),
                        ("antialiasing", {"on": True}, "AA full"),
                        ("textureOn", {"on": False}, "texture off"),
                        ("textureOn", {"on": True}, "texture on"),
                        ("textureMode", {"mode": "video"}, "texture video"),
                        ("videoTint", {"on": False}, "video tint off"),
                        ("reflectionLimit", {"iterations": 4}, "iterations 4"),
                        ("textureZoom", {"zoom": 2.5}, "zoom 2.5"),
                    ]
                    for name, args, expected in cases:
                        text(await session.call_tool(name, args))
                        assert expected in text(await session.call_tool("status", {})), name
                        callback_checks += 1
                    before_invalid = text(await session.call_tool("status", {}))
                    for name, args in [("animationOn", {"on": "false"}),
                                       ("symmetryType", {"symmetry": 1.5}),
                                       ("reflectionLimit", {"iterations": 41}),
                                       ("textureZoom", {"zoom": 0}),
                                       ("antialiasing", {"mode": "partial", "on": True}),
                                       ("antialiasing", {"mode": "true"}),
                                       ("tileColor", {"tile": "b"}),
                                       ("status", {"unexpected": True})]:
                        assert (await session.call_tool(name, args)).isError, name
                    assert text(await session.call_tool("status", {})) == before_invalid
                    text(await session.call_tool("reset", {}))
                    assert text(await session.call_tool("status", {})) == baseline
                for _ in range(100):
                    assert text(await session.call_tool("status", {})) == baseline
                # Within the 4095-byte UART request limit, but parser nodes exceed
                # the 32 KiB arena. A subsequent call must still succeed.
                exhausted = await session.call_tool("status", {"unused": [0] * 800})
                assert exhausted.isError and "arena exhausted" in exhausted.content[0].text
                assert text(await session.call_tool("status", {})) == baseline
                if task >= 4 and not no_vfs:
                    for _ in range(25):
                        resource_last = (await session.read_resource("hyperbolic://renderer/status")).contents[0].text
                        assert resource_last.startswith(baseline + "; frames ")
                    first_frame = int(re.search(r"; frames (\d+)", resource_first).group(1))
                    last_frame = int(re.search(r"; frames (\d+)", resource_last).group(1))
                    assert last_frame > first_frame
                    try:
                        await session.read_resource("hyperbolic://renderer/missing")
                    except McpError as exc:
                        assert exc.error.code == -32002
                    else:
                        raise AssertionError("Unknown URI must fail")
                    assert (await session.read_resource("hyperbolic://renderer/status")).contents[0].text.startswith(baseline)
                memory_after = text(await session.call_tool("mcpMemory", {}))
                assert re.search(r"heap allocations (\d+)", memory_after).group(1) == heaps
                if task >= 2:
                    assert int(re.search(r"tree height (\d+)", memory_after).group(1)) <= 6
                    assert int(re.search(r"lookup steps (\d+)", memory_after).group(1)) <= 6
                    unknown = await session.call_tool("unknown-tool", {})
                    assert unknown.isError
            finally:
                text(await session.call_tool("edgeColor", {"color": original_edge}))
            evidence = {"task": task, "tool_count": len(names), "status": baseline,
                        "memory_before": memory_before, "memory_after": memory_after,
                        "repeated_status_requests": 100, "invalid_input": "passed",
                        "arena_exhaustion_and_recovery": "passed", "heap_growth": 0,
                        "callback_mutation_checks": callback_checks,
                        "resource_first": resource_first, "resource_last": resource_last,
                        "vfs_enabled": not no_vfs if task >= 4 else None}
            Path(output).write_text(json.dumps(evidence, indent=2) + "\n")
            print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expect-no-vfs", action="store_true")
    args = parser.parse_args()
    anyio.run(run, args.task, args.url, args.output, args.expect_no_vfs)
