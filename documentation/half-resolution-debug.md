# Half-resolution crash investigation

## Live evidence (2026-09-15)

CMSIS Developer Assistant attached without reset to DevKit-E8 M55_HP.
Full resolution was running with CFSR/HFSR clear. Half resolution completed
several frames, then faulted during uninterrupted execution. At 100 kHz SWD,
the fault remained inspectable (an earlier run lost SWD access).

- UsageFault UNDEFINSTR at `et_pal_abort`, PC `0x8021e4e8`.
- This is the deliberate `__builtin_trap()` after an ExecuTorch assertion,
  not execution outside firmware: the image is in MRAM at `0x80200000`.
- Stack: `FreeableBuffer::data` -> `EthosUBackend::execute` ->
  `BackendDelegate::Execute` -> `Method::execute_instruction` ->
  `Method::execute` -> `app_main` line 732, frame 446.
- Backend ExecutionHandle: `0x200304a0`, processed pointer `0x0200b488`
  (`g_method_pool + 46216`).
- FreeableBuffer variant index at `0x0200b498` is 180, not valid 0 or 1.
- Buffer size is still 816. MSP is `0x200ffd28`, within the 32 KiB stack.
- A previous run returned Error::Ok from the upscaler and finished one whole
  half-resolution frame before later losing UART and debug access.

Corrupted buffer and following bytes:

```text
0200b480: 01 00 00 00 00 00 00 00 90 5f 22 80 a7 f2 27 e3
0200b490: 57 fc ff ff 74 69 2f 0b b4 78 00 02 01 00 00 00
0200b4a0: 00 00 00 00 30 03 00 00 3c 6c 00 20 a0 04 03 20
```

Root cause is not yet established. The last UART PC reported during a failed
SWD session was not enough to attribute the bug to the UART driver.

## Follow-up watchpoint attempt

A temporary debugger configuration watched byte `0x0200b498`. After a
debugger reset, it caught zero-initialization and the normal FreeableBuffer
move constructor, confirming the correct object address and an initial index
of zero. The first restarted run subsequently exited; its exit reason was
not captured. A second attempt captured the normal Ethos-U and renderer
startup banners, then the debugger became unresponsive during initialization.
The corrupting write has not yet been caught. The temporary reset/watchpoint
commands were removed from launch.json; the retained configuration attaches
without reset at 100 kHz SWD.

Next investigation: recover the VS Code debug session and board, attach at a
frame boundary, then arm a write watchpoint on the live buffer index after
initialization (avoiding startup watchpoint stops). If no CPU write triggers
before corruption, inspect NPU/DMA writes and cache maintenance. Do not assume
the buffer ownership, UART, or upscaler dimensions are at fault without that
evidence. No firmware fix or reflash has been performed for this investigation.

## Retry after VS Code recovery

Reattached to a running renderer with a valid buffer. Armed the same hardware
watchpoint after initialization, without reset. Half mode then ran for thousands
of frames without a watchpoint hit or buffer corruption. A later read still
showed index zero.

Since `hyperbolic_uart` tools were no longer exposed to this session, used the
CMSIS owned UART connection to exercise the board's existing JSON-RPC parser:
status replies succeeded, the original symmetry/colour settings were restored,
and all 40 alternating full/half `renderScale` requests (IDs 9100–9139) received
successful replies. Rendering continued afterwards. This exercises board MCP
dispatch but **does not reproduce the original end-to-end Hyperbolic bridge
tool invocation**. The user explicitly confirmed that bridge invocation was
the original trigger; the passing UART tests do not rule out that path.

Another `diagnose_fault` call stalled the debug connection even though a memory
read immediately before it showed a valid buffer. After debugger detachment,
UART logs showed rendering still progressing. Thus debugger-tool failures alone
must not be classified as firmware crashes.

Final state: restored full resolution with request 9200, acknowledged and
verified by multiple subsequent full-resolution frame reports (7075–7221).
Debugger detached, temporary watchpoint configuration removed, CMSIS serial
port closed. Actual bridge-tool reproduction remains pending bridge availability.

## Exact bridge path after reconnection

`hyperbolic_uart` became available again. With CMSIS debugging attached and a
write watchpoint on the valid variant index, the real bridge `renderScale`
tool successfully changed full to half. Eight more full/half cycles, each
followed by a bridge `status` call, also succeeded; the watchpoint did not stop
the target.

Then restarted firmware through the CMSIS debug pipeline, restored symmetry 1,
grey background and green tile A through the bridge, and issued the first half
request after initialization through `hyperbolic_uart.renderScale`. A fault
handler breakpoint did not fire. The subsequent bridge status succeeded. A
manual halt found the core in the geometry pass, and the buffer index was still
zero. This was a debugger firmware reset, not a board power cycle.

The original corruption remains unexplained; these passes are not a fix. No
firmware source was changed or rebuilt. Removed temporary debugger commands,
detached the debugger and restored full resolution through the actual bridge,
with a successful status reply confirming the requested settings.
