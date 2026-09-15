# Board layer: Alif Ensemble E8 DevKit (M55_HP + Ethos-U85)

Device: `Alif Semiconductor::AE822FA0E5597LS0:M55_HP` (Cortex-M55 high-performance
core, 400 MHz) with the Ethos-U85 (256 MACs) of the Ensemble E8.

Derived from the Ensemble pack's `Boards/DevKit-e8/Layers/M55_HP/Board_HP-U85.clayer.yml`
and trimmed to what the NPU render demo needs: headless inference plus the
MIPI DSI display. Camera, Ethernet, USB, VIO and vStream drivers are left out. The target-set in the
csolution debugs it through the on-board J-Link (`J-Link Server`, SWD at
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
**Terminal > Run Task > Alif: Install M55_HP debug stubs (DevKit-E8, single core configuration)**. The task needs
the VS Code setting `alif.setools.root` and SW4 in position SEUART. See
[the getting-started README](../../README.md).
