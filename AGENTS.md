NEVER install pyOCD. Use the one from the CMSIS vscode extension.

Use the CMSIS developer assistant MCP server to interact with the board.

Don't try to use pyOCD or gdb directly to interact with the board.
If the MCP server is returning errors and retrying is not solving the issue, tell the user that an action is required in vscode to solve the problem.

Serialize top-level CMSIS build/load/run/debug operations for this workspace and board. A tool timeout does not mean its underlying task stopped. Never repeat a launch merely to obtain status. Before starting another probe-owning operation, stop the existing CMSIS Run task or debug session and verify termination. If MCP cannot confirm Run-task termination, a read-only local process check may verify that the previous probe server has exited. If termination still cannot be established, request intervention in VS Code.

## Automatic flash/run workaround (Developer Assistant 2.3.9)

Use the following launch workflow through the CMSIS Developer Assistant MCP and the CMSIS Debugger extension's bundled tools. It can start a responsive session, but that alone does NOT verify that the intended firmware was flashed. Apply the image-verification requirements below as well.

1. Check `get_session_status`. Stop an existing debug session with `stop_debugging`, or a CMSIS Run task with `cmsis_action(action="stop_run")`, and verify termination as above.
2. Build and verify the selected images and generated cbuild-run configuration as described below. Then use `cmsis_action(action="load_and_debug", target="DevKit-E8@Release", timeoutMs=60000)`. The cbuild-run configuration must load both HP and HE Release images.
3. Poll `get_session_status` without launching again. The initial response may incorrectly say the session did not survive because it observed zero threads during startup; a subsequent status can report a responsive, running session. A launch response alone is not proof of failure.
4. Confirm `get_device_info` names the intended target and symbol file. `State: running` with `DAP responsive: true` establishes debug-session readiness only; it does not establish successful flashing or a healthy application. Verify load completion and application behavior below before reporting success. Leave the debugger attached for a requested running demo, and stop it before the next flash.

Preserve the workaround in `.vscode/launch.json`: the HP launch uses `run: "all"`, has no `tbreak main` in its initialization/reset commands, and sets `cmsis.updateConfiguration: "manual"`. This starts the application without a separate resume command. The manual launch currently references the Release image; keep its program path consistent with the selected target/build context.

Avoid `load_and_run` for completion tracking with Developer Assistant 2.3.9: its case-sensitive task filter misses uppercase `CMSIS Load`/`CMSIS Run` names, and Run intentionally starts a persistent server. The resulting "no task ran" response does not prove the board was not flashed. The standalone MCP `flash` tool also failed to resolve bundled pyOCD on PATH in this environment.

For unattended running, avoid `continue_execution`: on timeout it automatically pauses the target again. The MCP `reset(halt=false)` workaround also failed with this adapter because its evaluation request lacked a frame ID. Prefer the automatic launch configuration above.

## Flash the intended images and verify the result

Treat build completion, image programming, debugger attachment, and application startup as separate facts. Never report "the correct Release firmware is running" based only on a target label, the debugger's `program` path, or a responsive DAP session.

1. **Choose the target-set explicitly.** Use `DevKit-E8@Release` for the normal Release demo. Do not silently substitute Debug or Benchmark. If a diagnostic build is needed, state the change and keep both core images, generated configuration, and debugger symbols consistent with that build. Record temporary build-option changes, including enabling symbols in Release.
2. **Build before loading.** Build the chosen target-set through CMSIS MCP and obtain a successful cbuild completion. A timeout, an existing ELF, its timestamp, or "no task ran" is not proof of build success. A build may still be running after a tool returns; wait for its completion without starting overlapping builds or loads.
3. **Inspect the generated inputs after the build.** Follow `cmsis-executorch.cbuild-idx.yml` to the selected cbuild files and `out/cmsis-executorch+DevKit-E8.cbuild-run.yml`. Check `target-set`, both project contexts, and every image's path, `pname`, and `load` role. For Release, expect:

   | Core | Programmed image (`load: image`) | Debug symbols (`load: symbols`) |
   | --- | --- | --- |
   | M55_HP | `out/cmsis-executorch/DevKit-E8/Release/cmsis-executorch.hex` | `out/cmsis-executorch/DevKit-E8/Release/cmsis-executorch.axf` |
   | M55_HE | `out/M55_HE/DevKit-E8/Release/M55_HE.hex` | `out/M55_HE/DevKit-E8/Release/M55_HE.axf` |

   Paths inside cbuild-run are relative to that file. Resolve them before comparing. Do not mix an old HE image with a newly built HP image. Local artifact hashes can identify the intended build, but do not prove what is on the board.
4. **Check the actual load path.** The manual launch's `program` selects symbols; changing it does not select what CMSIS Load programs. Check that its `preLaunchTask`, the CMSIS Load task, and the selected cbuild-run all refer to the same intended build. An attach or "fault inspection (no reset)" configuration does not flash firmware. Preserve the automatic launch workaround above.
5. **Verify programming, not just launch dispatch.** Obtain CMSIS load output confirming the intended HP and HE image paths and successful load completion. A reply saying the pipeline was issued or is running is not completion evidence. Report byte verification only if the loader's verification or supported MCP readback actually establishes it; do not invoke pyOCD/GDB directly to fill this gap. If MCP cannot establish what was loaded, say that verification is incomplete and have the user complete/check the load in VS Code instead of declaring success or repeatedly launching it.
6. **Verify the application after launch.** Confirm the intended target/symbol file, then check that the renderer answers hyperbolic MCP status and that animation/video or frame progress is actually advancing. A black screen, a frozen frame, or a running debugger with an unresponsive UART is not a successful running demo. Distinguish confirmed programming from confirmed runtime behavior in the report.

If a user-performed VS Code reflash restores the display, treat that image as the working baseline. Do not overwrite it merely to repeat an unverified flash sequence. Investigate the prior image selection/load/startup sequence before attributing the failure to application code, a peripheral, or VS Code. A connection timeout does not by itself justify reloading VS Code; a reset-button restart and a complete power cycle are different operations, and recovery advice must reflect the evidence from the current occurrence.

Incident reminder (2026-09-16): the user restored animation by reflashing from VS Code after an assistant-led flash/debug investigation left a black screen. The exact cause was not established. Configured Release paths and a responsive debug session had been checked, but the programmed image had not been independently verified. Do not repeat that inference.

## For the demo

Styling, color, geometry, symmetry are related to the hyperbolic app running on board and the hyperbolic MCP server shoudl be used to control the app look and feel.
When I tell you to blink a color, it means alternating between this color and white or black quickly.
If the color is dark, you use white and if it is light you use black for the blinking.
