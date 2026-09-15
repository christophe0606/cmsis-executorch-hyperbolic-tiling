# Board layer: Alif Ensemble E8 DevKit (M55_HP + Ethos-U85)

Device: `Alif Semiconductor::AE822FA0E5597LS0:M55_HP` (Cortex-M55 high-performance
core, 400 MHz) with the Ethos-U85 (256 MACs) of the Ensemble E8.

Derived from the Ensemble pack's `Boards/DevKit-e8/Layers/M55_HP/Board_HP-U85.clayer.yml`
and trimmed to what the NPU render demo needs: headless inference plus the
MIPI DSI display. Camera, Ethernet, USB, VIO and vStream drivers are left out. The target-set in the
csolution debugs it through ULINKplus (`ULINKplus@pyOCD`, SWD at
4 MHz, `start-pname: M55_HP`).

| File | Purpose |
|------|---------|
| `Board-U85.clayer.yml` | Layer: startup, SE services, UART4 stdio, Ethos-U85 driver, memory placement |
| `main.c` | Pin/GPIO config, SE services, clocks, MIPI DPHY power, stdio, NPU init, then `app_main()` |
| `board_display.c`, `board_display.h` | CDC200 display controller bring-up (ILI9806E 480x800 panel over MIPI DSI), vsync-synchronised frame buffer switch, start-of-frame counter |
| `retarget_stdio.c`, `board_console.h` | UART4 at 115200 8N1, interrupt receive ring and stdio character hooks; see [MCP over UART](../../documentation/mcp-uart.md) |
| `ethos_setup.c` | Ethos-U85 driver init at `NPU_HG_BASE`, IRQ 366, prints the NPU banner |
| `ethosu_cb_dcache.c` | D-cache clean/invalidate hooks for NPU buffers outside the TCMs |
| `linker_ac6_mram.sct.src`, `linker_gnu_mram.ld.src` | Pack linker scripts plus a 32 kB stack and the `.bss.ai_pool` section in bulk SRAM |
| `RTE/` | Configuration files carried with the layer (see below) |

## Memory layout

| Region | Address | Used for |
|--------|---------|----------|
| MRAM (HP application region) | `0x80200000`, 2 MB | Code, constants, the embedded `.pte` model |
| DTCM (SRAM3) | `0x20000000` (core alias; `0x50800000` global), 1 MB | `.data`/`.bss` (including the runner's 672 kB int8 G-buffer), 96 kB heap, 32 kB stack |
| SRAM0/SRAM1 (bulk) | `0x02000000`, 8 MB combined (`SRAM0_SRAM1_COMBINED` in `app_mem_regions.h`; the GNU script uses SRAM0 alone, 4 MB, which this demo no longer fits) | `.bss.ai_pool`: the runner's 1.5 MB method pool, 3.5 MB temp pool (NPU scratch; Vela needs 3.0 MB for the 480x800 shade graph) and 192 kB z-buffer; `.bss.lcd_frame_buf`: two 1.15 MB RGB888 frame buffers |

The pool sizes and sections come from the `define:` node of the layer
(`APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION`,
`APP_FRAMEBUFFER_SECTION`) and are consumed by `src/app_main.cpp`, as are
`APP_HAS_DISPLAY` and `APP_DISPLAY_WIDTH/HEIGHT`.

## Dual-core memory ownership

The DevKit-E8 target-set loads two independently linked projects. HP runs
`cmsis-executorch`; `M55_HE` runs the pack's HE startup and a Helium worker.
HE has no board, display, UART, NPU or Secure Enclave service setup.

| Memory | HP ownership | HE ownership |
|--------|--------------|--------------|
| Shared MRAM | `0x80200000`–`0x803FFFFF` | `0x80000000`–`0x801FFFFF` |
| Local DTCM | `0x20000000`–`0x200FFFFF` (global SRAM3 at `0x50800000`) | `0x20000000`–`0x2003FFFF` (global SRAM5 at `0x58800000`) |
| Bulk SRAM0/SRAM1 below `0x027E0000` | All application pools and frame buffers | No allocations |
| Shared mailbox, `0x027E0000`–`0x027FFFFF` | Request and immutable frame inputs | Response and completed strip |
| MRAM user area, `0x80400000`–`0x8057FFFF` | Available to HP's linker | No allocations |

The identical local TCM addresses refer to different physical memories on
the two cores. Shared MRAM and bulk SRAM do not have that property: independent
linkers cannot detect conflicts with another executable. They must follow an
agreed memory partition. Merely naming a region in a script does not consume
it; the map file shows which sections actually occupy it.

Both linker configurations read
`RTE/Device/AE822FA0E5597LS0_M55_HP/app_mem_regions.h`. HE's startup configuration
includes that same header, so the region definitions stay consistent.
`M55_HE/linker_ac6_mram.sct.src` limits HE's code/data to `APP_MRAM_HE_SIZE`
and local TCM, and checks the MRAM partition boundaries. Both AC6 scripts
reserve the mailbox as an `EMPTY` region (no startup initialization).
The HP pool region ends before it; the MPU still maps the entire 8 MB SRAM.

### Parallel Helium rendering

`src/tiling_kernel.cpp` is compiled with speed optimization into both images.
It reflects, shades and antialiases one 240×16 strip using each core's local
DTCM state and output buffer. HP maintains the queue, renders strips while HE
is busy, copies completed HE results back to HP DTCM, and immediately sends
HE another strip. This lets the 400 MHz HP take more work than the 160 MHz HE.
HP alone converts results to framebuffer pixels and calls Ethos for half-scale
upscaling, including the existing halo rows. Completion order does not affect
pixel placement, and a frame is presented only after every strip is finished.

`src/tiling_worker.cpp` publishes a settings/maps/texture snapshot once per
frame. Request and response occupy separate 32-byte cache lines, each with
one writer. Payload clean/barrier operations precede sequence publication;
the receiver invalidates before reading. No local TCM pointer crosses cores.
HP waits for a startup handshake before writing frame inputs. If HE is absent,
incompatible or times out, HP completes rendering locally and stops issuing
requests; late HE writes remain confined to the reserved mailbox. Load and
reset both images together when changing the protocol or shared structures.

At startup, 32 strip comparisons check HP/HE pixels and reflection statistics
across full/half, AA on/off, texture on/off and disk/plane modes, including
the center and clamped bottom rows. These use private test settings and leave
the displayed settings unchanged. A failure disables offloading. The console
reports validation results and HE's strip count/transfer-wait time.
`g_tiling_metrics` exposes frame timings and validation results to the debugger;
`g_tiling_use_he=0` selects HP-only rendering for comparison. `render_cycles`
measures elapsed rendering time including mailbox work, excluding display wait
and console output. `geometry_cycles` measures HP geometry and HE result waits,
not the sum of the two cores' simultaneous execution times.

Measured on the E8 with AC6 6.24, HP at 400 MHz and HE at 160 MHz, full
resolution, no AA, texture enabled, 12 reflection rounds and animation running:

| Mode | Sampled render times | HE strips per 100-strip frame |
|------|----------------------|--------------------------------|
| HP only (`g_tiling_use_he=0`) | 83.08 ms, 81.68 ms | 0 |
| HP + HE (`g_tiling_use_he=1`) | 65.91 ms, 63.51 ms | 24, 25 |

These samples indicate roughly 25–30% more rendering throughput (about
12 to 15–16 rendered frames/s), not a measured doubling of LCD frame rate.
They are different animation frames with identical settings; display waits
and console overhead are excluded. HE result transfers/waits took 0.82–0.85 ms
in the dual-core samples. All 32 startup comparisons passed. Both firmware
images built successfully and were loaded using the CMSIS extension's bundled
debugger/pyOCD. The renderer was left with dual-core processing enabled.

Build with the CMSIS panel or
`cbuild cmsis-executorch.csolution.yml --active DevKit-E8 --update-rte`.
The generated `out/cmsis-executorch+DevKit-E8.cbuild-run.yml` associates each
ELF/HEX with its processor and loads both images. HP remains the initial
debug core on port 3333; HE uses port 3334. If its symbols are missing, use
**CMSIS Solution → Manage Solution → Debugger → Apply** to refresh the launch
configurations after adding the project.

## Display

CDC200 → MIPI DSI (2 lanes) → ILI9806E panel, configured in this layer's
`RTE_Device.h` (the pack's DevKit-E8 settings): RGB888, 480x800, 60 fps,
panel variant 1. `main.c` powers the MIPI DPHY through the VBAT power
control register before `stdio_init()`, as the pack's layer does;
`board_display.c` drives the CDC200 driver.

## RTE configuration

Unlike the Corstone-320 layer, `RTE/` is committed for this layer (see
`.gitignore`): Alif's stdio retarget refuses to build unless UART4 is in
polling mode; the local retarget now replaces it and uses interrupt mode
(`RTE_UART4_BLOCKING_MODE_ENABLE 0` in `RTE_Device.h`). The
Conductor-generated `pins.h`/`board_defs.h` differ from the pack defaults. The
files are the pack's own DevKit-E8 layer configuration.

## STDIO

UART4 on the PRG USB connector, 115200 baud. Set **SW4 to UART4** for the
console; SW4 in position **SEUART** (the default) routes the same USB serial
port to the Secure Enclave for SETOOLS. The on-board J-Link sits behind the
same connector.

## Before the first debug session

Program the debug stubs into the device's ATOC with Alif SETOOLS once:
**Terminal > Run Task > Alif: Install M55_HP and M55_HE debug stubs (DevKit-E8, dual core configuration)**. The task needs
the VS Code setting `alif.setools.root` and SW4 in position SEUART. See
[the getting-started README](../../README.md).
