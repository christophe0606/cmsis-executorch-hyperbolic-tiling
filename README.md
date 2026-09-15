# ExecuTorch on Ethos-U85: hackathon guide

The current Helium renderer supports **full resolution, 2x2 antialiasing and
an adjustable reflection limit**. It defaults to full resolution, AA off and
12 rounds. Use `scale half` for faster rendering with Ethos enlargement.
See [quality controls and measured board results](documentation/helium-quality.md).

Plane geometry is rotated 90 degrees to fill the portrait display. Use
`texture off` (MCP: `textureOn(on=false)`) for solid red/blue tiles, or change
their colours with `tile a|b <colour>`. `texture on` restores texture blending.

**MCP over UART is available without an RTOS.** See
[UART MCP and Codex workspace configuration](documentation/mcp-uart.md), including
the serial bridge and the `.codex/config.toml` example.

The Arm ExecuTorch example, on its `hackathon` branch with the Alif board
added. One CMSIS solution runs an ExecuTorch program on the Ethos-U85 of the **Alif Ensemble E8 DevKit** (Cortex-M55
HP core) and on the **Corstone-320 FVP**; you switch between them by
target-type. On this branch the program is the **hyperbolic tiling**: a port
of christophe0606's GLSL shader demo, with the reflection geometry and the
texture gather, compositing and antialiasing on the Cortex-M55 with Helium,
half-resolution upscaling on the NPU, the DevKit's LCD showing it and the demo's MCP tools over UART
alongside console commands; see [documentation/hyperbolic-tiling.md](documentation/hyperbolic-tiling.md).
The `npu-render` branch has the 3D-pipeline demo this builds on
([documentation/npu-render.md](documentation/npu-render.md)). The model is exported from PyTorch in three steps: the
CMSIS-Toolbox describes the target, `create_ai_layer.py` turns that into the
AI layer, the toolbox builds the application. This page takes you from an
empty machine to a debug session on the board. Everything about the example
itself is in [documentation/example.md](documentation/example.md).

## 1. Host tools

1. Install [VS Code](https://code.visualstudio.com/).
2. Install the extensions **Keil Studio Pack** (`Arm.keil-studio-pack`, which
   brings the CMSIS Solution extension 1.70.0 or newer that the project's
   task drop-ins need) and **Python** (`ms-python.python`). Sign in with an
   Arm account when Keil Studio asks; the free Keil MDK Community license is
   enough.
3. Nothing else by hand: when you open the project, the Arm Tools Environment
   Manager offers to install the tools pinned in `vcpkg-configuration.json`
   (CMSIS-Toolbox, Arm Compiler 6, GCC, CMake, Ninja, and on Linux and
   Windows the Corstone-320 FVP; macOS runs it in Docker, see step 8). Accept.
4. Optional: the **CMSIS Developer Assistant** extension lets an AI agent
   (Claude Code or GitHub Copilot Chat) build, flash and debug the board
   through an MCP server. Install it, install one of the agents, run
   **CMSIS Developer Assistant: Configure Agents and Skills** from the command
   palette and pick at least the `cmsis-debug-live` and `cmsis-help` skills.

## 2. Alif and SEGGER tools (board only)

1. **Alif SETOOLS** V1.110.000 or later from the
   [Alif software and tools page](https://alifsemi.com/support/software-tools/ensemble/)
   (login required). Unpack it; on Linux and macOS make the tools executable
   and install the Python packages its README lists. Add the root directory
   (the one with `app-gen-toc` and `app-write-mram`) to your VS Code user
   settings:

   ```json
   "alif.setools.root": "/absolute/path/to/setools"
   ```

2. **SEGGER J-Link Software** V8.42 or later from
   [segger.com](https://www.segger.com/downloads/jlink/). The board has an
   on-board J-Link.

## 3. Board

- Connect a USB-C cable to **PRG USB** (the connector in the corner). It
  powers the board and carries the J-Link and a USB-to-UART bridge. Leave
  **MCU USB** unconnected.
- Jumpers at their defaults: **JP5 on 1-2**, **JP7 on 3-4**. Never move
  jumpers with power applied.
- **SW4** selects what the UART bridge is connected to:

  | SW4 | Connected to | Used for |
  |-----|--------------|----------|
  | `SEUART` (default) | Secure Enclave UART | SETOOLS (step 5) |
  | `UART4` | Application UART4, 115200 8N1 | The example's console (step 6) |

- With the board attached, run these once: in the SETOOLS directory
  `updateSystemPackage -d` (SW4 on `SEUART`; picks the serial port, checks the
  system firmware, offers to make the E8 the default target: answer yes), and
  J-Link Commander (`JLinkExe`, `JLink.exe` on Windows), which updates the
  on-board J-Link firmware and installs its serial-port drivers.

## 4. Project

```bash
git clone https://github.com/Arm-Examples/CMSIS-Executorch.git
cd CMSIS-Executorch
git checkout hackathon
```

Open the folder in VS Code and accept the tool activation and the pack
installation (`PyTorch::ExecuTorch`, `AlifSemiconductor::Ensemble`, CMSIS).
In the CMSIS view open **Manage Solution**, choose the target-type
**DevKit-E8** (or **SSE-320-U85** for the FVP) and click **Apply**.

The repository ships a generated AI layer, so no Python is needed to build.
To change the model, run **Terminal > Run Task > Setup Python virtual
environment** once (several GB of PyTorch, takes a while; the **(uv)**
variant of the task uses uv and can download the Python version it asks
for), edit `model/model.py`, and run the task **Create AI layer** before
building.

## 5. Prepare the board once

The Secure Enclave boots the M55 cores from a table of contents in MRAM; the
debugger needs that table to point at a debug stub.

1. SW4 to **SEUART**, PRG USB attached.
2. **Terminal > Run Task > Alif: Install M55_HP and M55_HE debug stubs (DevKit-E8, dual core configuration)**. Choose COM port
   discovery (`-d`) the first time; SETOOLS remembers the port. The task
   copies the configuration and stub from `.alif/` into the SETOOLS tree and
   runs `app-gen-toc` and `app-write-mram`.
3. SW4 to **UART4**.

Repeat this after another project has reprogrammed the table.

This solution uses both cores. The task installs both MRAM debug stubs using
`.alif/M55_HP_HE_mram_cfg.json` (HP at `0x80200000`, HE at `0x80000000`).
The dual core configuration and HE stub come from the DevKit-e8 DualCore
example in `AlifSemiconductor::Ensemble@2.2.1`; the existing HP stub is
identical to that example's HP stub.

The DevKit-E8 target builds two projects: `cmsis-executorch` runs the
hyperbolic app on HP; `M55_HE` runs a Helium strip-rendering worker. HP owns
the dynamic strip queue, Ethos upscaling and the display. Both images are
included in the target-set and must be loaded together.
The HE project is excluded from the SSE-320-U85 simulator target.
See [the dual-core memory layout](board/DevKit-E8/README.md#dual-core-memory-ownership)
for why their linker scripts do not conflict.

## 6. Build, run, debug

For an optimized build, select the **Release** target-set under **DevKit-E8**
in Manage Solution. It selects Release for both HP and HE. From a CMSIS
toolchain terminal:

```sh
cbuild cmsis-executorch.csolution.yml --active DevKit-E8@Release --update-rte
```

Release uses `-O3` throughout and `-ffast-math` for the Helium kernel on both
cores, with debug information disabled. The default target-set retains Debug;
its application groups already use `-O3`. The simulator also has a Release
target-set (`SSE-320-U85@Release`). Release outputs are in each project's
`out/<project>/<target>/Release/` directory. Select the corresponding target-set
before loading so the generated load configuration uses the intended images.

1. With SW4 on **UART4**, open the **Serial Monitor** panel on the PRG USB
   port, 115200 baud.
2. In the CMSIS view click **Build**, then **Debug** (or **Run**). Keil Studio
   starts the CMSIS Debugger with ULINKplus/pyOCD over SWD, loads both images into MRAM and stops
   at `main`; continue with F5. The console shows:

   ```text
   Ethos-U version info:
       Arch:       v2.0.0
       MACs/cc:    256
       Cmd stream: v1
   NPU render: Ethos-U85 as a tensor coprocessor for a 3D pipeline, Helium on the CPU
     program: 7696 bytes, 2 objects x 512 vertices, G-buffer 240x400 int8, frame 480x800 RGB888
     scene: torus 512 + sphere 480 vertices, 1920 triangles
     display: 480x800 RGB888 on (status 0)
   frames 0..119 avg: vertex-NPU 148 us (copy 43) | post-vertex 182 us | raster 9766 us (...) | shade-NPU 8118 us (frame copy 2531, vsync wait 2) | check 1752 us | frame 18.9 ms = 53.0 fps | max err 1.6/255
   ```

   and the board's LCD shows a lit, fogged torus with a sphere orbiting it,
   shaded and upscaled by the NPU, rasterized by the M55 with Helium.

3. Set a breakpoint after `module.execute(MODEL_SHADE_METHOD, ...)` in
   `src/app_main.cpp` and inspect the G-buffer or the frame buffer, or ask
   the CMSIS Developer Assistant to do it: "Build for the DevKit-E8, load it,
   break after the shade method and read a row of the back frame buffer."

**FVP instead of the board:** choose the **SSE-320-U85** target-type and click
**Run** or **Debug**; the same output appears in the terminal. On macOS the
FVP runs in Docker (Docker Desktop must be running; the first run builds the
image, about 100 MB). On Windows set `model:` in the csolution's SSE-320-U85
target-set to `FVP_Corstone_SSE-320`.

## 7. If something does not work

- **J-Link connects but never stops at `main`:** the table of contents does
  not point at the debug stub. Repeat step 5.
- **Debug hangs at "Connecting":** the target-set was switched to
  `protocol: jtag`. Keep SWD: the generated load task blocks on JLinkExe's
  JTAG-chain prompt, and the device stays in SWD mode after any SWD use
  until it is power-cycled.
- **No console output:** SW4 is still on `SEUART`, or the port was opened
  before the switch was moved. Set `UART4` and reopen the port.
- **The board hard-faults right after a flash, before printing anything:**
  the J-Link loader (`CMSIS Load`, `JLinkExe LoadFile`) can corrupt the
  first 16 bytes of the image in MRAM and report `Programming failed @
  address 0x80200004 (block verification error)`; the reset vector then reads
  as code bytes. Program with pyOCD instead (`pyocd load --cbuild-run
  out/cmsis-executorch+DevKit-E8.cbuild-run.yml`, or the CMSIS Developer
  Assistant's flash tool), which uses the pack's flash algorithm and leaves
  the Secure Enclave's table of contents untouched, then Debug or reset.
- **`app-write-mram` gets no answer:** press reset while it waits, check SW4
  is on `SEUART`, close any terminal holding the port.
- **"torch is not installed":** run the task **Setup Python virtual
  environment** first; it is only needed to regenerate the AI layer.
- After **Apply**, the extension adds a J-Link entry to `.vscode/launch.json`
  next to the committed FVP entry. That is expected.

## Read on

- [documentation/example.md](documentation/example.md): the example, the
  command-line flow, how the model is generated.
- [board/DevKit-E8/README.md](board/DevKit-E8/README.md): the board layer,
  memory map and RTE configuration.
- [documentation/mlops-flow.md](documentation/mlops-flow.md): the `mlops:`
  node and `*.cbuild-mlops.yml` in detail.
- [documentation/pack-provenance.md](documentation/pack-provenance.md): where
  the `PyTorch::ExecuTorch` pack comes from and how to move to a new version.
