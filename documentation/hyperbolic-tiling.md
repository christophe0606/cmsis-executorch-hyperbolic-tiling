# Hyperbolic tiling: a fragment shader split between Helium and the Ethos-U85

The current renderer adds [full resolution, 2x2 antialiasing and an adjustable
reflection budget](helium-quality.md). That page describes the current controls
and CPU/NPU split. The implementation and measurements below describe the
original half-resolution pipeline used as the baseline.

Branch `hyperbolic-npu`. A port of
[christophe0606/shader_linux_glsl](https://github.com/christophe0606/shader_linux_glsl),
Christophe Favergeon's full-screen GLSL shader that tiles the Poincaré disk
with reflected copies of a camera image and is steered by an LLM through
MCP tools, onto the Alif Ensemble E8 DevKit: Cortex-M55 with Helium plus
Ethos-U85, output on the 480x800 panel. It reuses the NPU render
infrastructure of the `npu-render` branch (multi-method export with
quantized IO, the board layer with the display, the frame-buffer routing).

## The split

The original does everything per pixel in one fragment shader. That shader
has exactly the shape the NPU cannot run: a loop of up to 40 rounds of
three `if (hdot(p, n) < 0)` reflections, a data-dependent exit, and a
texture read at a computed coordinate. So:

| Stage | Where | What |
|-------|-------|------|
| Möbius animation, reflections into the fundamental triangle with early exit, reflection count (tile parity), edge-distance test, Poincaré coordinates, texture coordinate, texture lookup | CPU, Helium, four pixels per vector | writes a 240x400 "tiling G-buffer": the gathered texel (3 int8 planes), parity, edge and inside masks (bool) |
| Tile A/B colouring, edge and background compositing, 2x bilinear upscale, NHWC transpose | NPU, one ExecuTorch method `tile` in [`model/model.py`](../model/model.py) | 800x480x3 RGB888 frame, copied straight into the back frame buffer |

The reflection loop is written branch-free per lane, as the shader's own
`min(0, hdot)` formulation already is: a predicated counter tracks the
number of reflections, and a vector iterates until no lane moved in a
round. Rounds are counted and reported. Vector reciprocals (MVE has no
divide) use the bit-hack seed and two Newton steps; `fract` uses `vrndm`.

The texture read stays on the CPU as Helium gather loads (`vldrb` with
vector offsets, one per colour plane, four texels per instruction), so the
G-buffer carries the texel rather than its index. The NPU version, a
`torch.index_select` on the texture that the Arm backend lowers to TOSA
GATHER, compiled with Vela and ran on the U85, but returned the wrong
texels on the board: with a constant index every pixel still got a
different, per-position grey value, in both the row-major and the planar
texture layout. That is left as an open question about the backend's
gather lowering; the Helium gather costs well under a millisecond.

The compositing uses `torch.where` selects on bool masks rather than
`mask * a + (1 - mask) * b`: the multiply-and-subtract chain lost up to
13 % at full brightness once quantized to int8 (reproduced on the host with
the fake-quantized graph), the selects are exact. The exporter also
calibrates int8 activations with min/max observers instead of the Arm
quantizer's histogram observer, which clipped the pinned [0, 1] ranges.
Vela reports 11 NPU operators, 0 CPU operators, 2.6 MB of scratch. Tile,
edge and background colours are method inputs, so the console commands
change them without a re-export.

## MCP tools, console commands and the camera

The original's MCP tools now run over UART using an embedded `c_mcp` port,
interrupt reception and dispatch between frames; no RTOS is required. See
[MCP connection and Codex configuration](mcp-uart.md). The same UART console
(115200 on PRG USB) also accepts these commands from a terminal or the CMSIS
Developer Assistant's serial tools:

```text
symmetry 0|1|2          (2,4,5), (2,4,7) or (4,4,4) triangle group, as the demo's presets
geometry disk|plane     Poincaré disk or vertical strip filling the portrait panel
animation on|off        the Möbius drift
edge <colour>           edge colour: a name (black, white, red, ... navy) or r,g,b
background <colour>
tile a|b <colour>       the two tile colours (default red and blue, as the shader)
texture on|off|video    procedural blend, solid tile colours, or live camera (default on)
video-tint on|off       blend video with tile colours or show original camera colours (default on)
zoom <f>                texture zoom
reset | status | help
```

There is no camera on the board build, so a procedural 128x128 texture that
drifts with time stands in for the video frames. The pack's CPI/CSI2
camera drivers and vStream VideoIn are the path to a real one.

## Files

- `model/model.py`: the `tile` method and its calibration.
- `src/app_main.cpp`: geometry pass, texture, console, timing, reference check.
- `create_ai_layer.py`: integer inputs (the gather index) pass through the
  quantized-IO export unquantized.
- `board/DevKit-E8/Board-U85.clayer.yml`: temp pool sized for the tile graph.

## Running it

```bash
cbuild setup cmsis-executorch.csolution.yml --active DevKit-E8
python3 create_ai_layer.py cmsis-executorch.cbuild-mlops.yml
cbuild cmsis-executorch.csolution.yml --active DevKit-E8
pyocd load --cbuild-run out/cmsis-executorch+DevKit-E8.cbuild-run.yml   # see the README on the J-Link loader
```

Every 120 frames the console prints the geometry time with the average
number of reflection rounds per vector, the NPU time with its frame copy,
the frame rate and the worst deviation of the NPU frame from a float
reference of the composition on a 12x20 pixel grid. `preview` prints the
shown frame as ASCII, `probe x y` one pixel with its G-buffer entry.

## Measured on the DevKit-E8

| Stage | Where | Time |
|-------|-------|------|
| Geometry pass, 96 000 pixels, ~3 reflection rounds per vector on average with early exit, including the texture gather | CPU, Helium | 35 ms |
| Texture scroll | CPU | 0.1 ms |
| Compositing + 2x bilinear upscale to 480x800 | NPU | 7.6 ms, 2.5 ms of it the frame copy |
| Frame | | 44 ms, 23 fps |

The NPU frame matches the float reference within 1/255 across the
presets. The geometry pass runs at about 2.6 cycles per vector instruction,
which is the Cortex-M55's Helium issue rate for this mix of float
multiply-accumulates and predicated operations; a faster frame needs fewer
pixels (a 120x200 geometry with a 4x upscale) rather than a tighter loop.
The original shader also supersamples 4x4 per pixel; this port relies on
the bilinear upscale instead.
