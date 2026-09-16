# Video plane-to-disk crash investigation

2026-09-16, source revision `ab08de0`, DevKit-E8, CMSIS Developer Assistant 2.3.9.

## Status

The immediate plane-to-disk crash has **not been isolated or fixed**. A later
visible freeze with failed UART/debug access occurred during this investigation.
The user recalled half resolution, partial antialiasing, probably symmetry 2,
and untinted video for the original occurrence.

The original session could not read core registers or fault memory. Its lone
`camera_callback` frame was not corroborated after reconnecting and is not
reliable evidence of the crash location. A no-reset reconnect also failed to
read registers. No fault registers were recovered from that occurrence.

A reset-button restart restored debug access but left display initialization
waiting in `DPHY_MasterSetup` for the display PLL. `g_tiling_metrics.frames`
was zero, and CFSR/HFSR were clear. A subsequent full power cycle allowed the
Debug image to boot and respond to MCP. This startup issue is distinct from
the unconfirmed cause of the original crash.

## On-board checks

All appearance changes were issued through the hyperbolic UART MCP server.
The bridge successfully separates normal Debug trace lines from JSON-RPC
replies.

| Build | Plane-to-disk transitions | Settings |
| --- | ---: | --- |
| Debug | 9 | Full resolution, partial AA, symmetry 0, tinted video |
| Debug | 18 | Full/half resolution, all AA modes and all three symmetries, tinted video |
| Debug | 12 | Half resolution, partial AA, symmetry 2, untinted video |
| Release with symbols | 12 | Half resolution, partial AA, symmetry 2, untinted video; one-second dwell in each geometry |
| Release with symbols | 15 | Half resolution, partial AA, all three symmetries, untinted video, 40 reflection rounds; half-second dwell |

All 66 transitions initially remained responsive. Release retains speed optimization
and the existing `-ffast-math` kernel/DSP flags; enabling symbols does not
enable `APP_FRAME_PERF_LOG`. The Release CMSIS build reported exit code 0.

At the last successful Release inspection, the frame counter was 1479,
`validation_errors` was zero, and HE completed 9 strips in the sampled frame.
Camera state was `active=1`, `camera_events=1`, `stop_error=0`.
The fault decoder reported clear CFSR/HFSR. These observations establish
successful execution during the checks, not absence of an intermittent bug.

After that successful inspection, HP was resumed. A later diagnostic pause
reported a lone `cdc_callback` frame, but both frame-counter and camera-timestamp
evaluations returned blank values. After a resume request, UART status timed
out at 15 seconds. The user confirmed the display was frozen. The adapter
reported a running, responsive DAP session, but a subsequent 15-second pause
request could not halt the target. No new fault registers were recovered.
The timing does not distinguish a spontaneous video failure from a failure
associated with debugger halt/resume; it does not establish that the geometry
transition itself caused this occurrence. The HP session was stopped before
attempting a no-reset HE inspection.

The HE inspection request returned no response from the VS Code window within
75 seconds. MCP reported no session, and a read-only process check found no
remaining probe server or GDB process. The temporary HE launch was removed;
the attempt establishes no facts about HE's hardware state. Reload the VS Code
window before further CMSIS debugging, then power-cycle the board and restart
the UART bridge for a fresh reproduction. The bridge deliberately refuses
further exchanges after its timed-out request until restarted.

## Capturing a recurrence

Release source symbols are enabled in the solution, and the manual no-reset
inspection launch now references the Release ELF. The regular automatic
launch workaround remains intact. HP fault vector catch was enabled through
CMSIS MCP by preserving DEMCR and setting mask `0x7f0` (original value
`0x01100400`, with catches `0x011007f0`). This register setting is session
state, not a firmware change.

The settings at the later freeze were half resolution, partial AA, symmetry 2,
12 reflection rounds, animation on, disk geometry, and untinted live video.
For a subsequent recurrence, inspect
the stopped state before resetting: session status, call stack, fault
decoder, frame/worker metrics, and camera state. Follow `AGENTS.md` when
stopping/reconnecting; never launch overlapping probe sessions.
