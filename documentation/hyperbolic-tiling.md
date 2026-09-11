# Hyperbolic tiling: a fragment shader split between Helium and the Ethos-U85

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
| Möbius animation, reflections into the fundamental triangle with early exit, reflection count (tile parity), edge-distance test, Poincaré coordinates, texture coordinate | CPU, Helium, four pixels per vector | writes a 240x400 "tiling G-buffer": texel index (int32), parity, edge and inside masks (int8) |
| Texture lookup (gather), tile A/B colouring, edge and background blending, 2x bilinear upscale, NHWC transpose | NPU, one ExecuTorch method `tile` in [`model/model.py`](../model/model.py) | 800x480x3 RGB888 frame, copied straight into the back frame buffer |

The reflection loop is written branch-free per lane, as the shader's own
`min(0, hdot)` formulation already is: a predicated counter tracks the
number of reflections, and a vector iterates until no lane moved in a
round. Rounds are counted and reported. Vector reciprocals (MVE has no
divide) use the bit-hack seed and two Newton steps; `fract` uses `vrndm`.

The texture read is `torch.index_select` on a `(128*128, 3)` texture with a
rank-1 int32 index, which the Arm backend lowers to TOSA GATHER; the U85
runs it natively. Vela reports 13 NPU operators, 0 CPU operators, 2.6 MB of
scratch. Tile, edge and background colours are method inputs, so the
console commands change them without a re-export.

## What replaces the MCP tools and the camera

The original's MCP tools become commands on the UART console (115200 on the
PRG USB port), so a terminal or the CMSIS Developer Assistant's serial tools
play the LLM's part:

```text
symmetry 0|1|2          (2,4,5), (2,4,7) or (4,4,4) triangle group, as the demo's presets
geometry disk|plane     Poincaré disk or the strip model of the plane
animation on|off        the Möbius drift
edge <colour>           edge colour: a name (black, white, red, ... navy) or r,g,b
background <colour>
tile a|b <colour>       the two tile colours (default red and blue, as the shader)
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
reference of the composition on a 12x20 pixel grid.
