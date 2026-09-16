NEVER install pyOCD. Use the one from the CMSIS vscode extension.

Use the CMSIS developer assistant MCP server to interact with the board.

Don't try to use pyOCD or gdb directly to interact with the board.
If the MCP server is returning errors and retrying is not solving the issue, tell the user that an action is required in vscode to solve the problem.

Serialize top-level CMSIS build/load/run/debug operations for this workspace and board. A tool timeout does not mean its underlying task stopped. Never repeat a launch merely to obtain status. Before starting another probe-owning operation, stop the existing CMSIS Run task or debug session and verify termination. If MCP cannot confirm Run-task termination, a read-only local process check may verify that the previous probe server has exited. If termination still cannot be established, request intervention in VS Code.

## Automatic flash/run workaround (Developer Assistant 2.3.9)

Use the following workflow through the CMSIS Developer Assistant MCP. It was verified with DevKit-E8@Release and the CMSIS Debugger extension's bundled tools.

1. Check `get_session_status`. Stop an existing debug session with `stop_debugging`, or a CMSIS Run task with `cmsis_action(action="stop_run")`, and verify termination as above.
2. Use `cmsis_action(action="load_and_debug", target="DevKit-E8@Release", timeoutMs=60000)`. The cbuild-run configuration loads both HP and HE Release images.
3. Poll `get_session_status` without launching again. The initial response may incorrectly say the session did not survive because it observed zero threads during startup; a subsequent status can report a responsive, running session. A launch response alone is not proof of failure.
4. Confirm `get_device_info` names the intended target and image. Success is `State: running` with `DAP responsive: true`. Leave the debugger attached for a requested running demo, and stop it before the next flash.

Preserve the workaround in `.vscode/launch.json`: the HP launch uses `run: "all"`, has no `tbreak main` in its initialization/reset commands, and sets `cmsis.updateConfiguration: "manual"`. This starts the application without a separate resume command. The manual launch currently references the Release image; keep its program path consistent with the selected target/build context.

Avoid `load_and_run` for completion tracking with Developer Assistant 2.3.9: its case-sensitive task filter misses uppercase `CMSIS Load`/`CMSIS Run` names, and Run intentionally starts a persistent server. The resulting "no task ran" response does not prove the board was not flashed. The standalone MCP `flash` tool also failed to resolve bundled pyOCD on PATH in this environment.

For unattended running, avoid `continue_execution`: on timeout it automatically pauses the target again. The MCP `reset(halt=false)` workaround also failed with this adapter because its evaluation request lacked a frame ID. Prefer the automatic launch configuration above.


Styling, color, geometry, symmetry are related to the hyperbolic app running on board and the hyperbolic MCP server shoudl be used to control the app look and feel.
