# NPU render: the Ethos-U85 as a tensor coprocessor for a 3D pipeline

A prototype of the idea that the Ethos-U85 is not a 3D accelerator but is
usable for the parts of a 3D pipeline that are plain tensor math. It runs on
the Alif Ensemble E8 DevKit (Cortex-M55 HP core at 400 MHz with Helium +
Ethos-U85 with 256 MACs) through the same csolution, `create_ai_layer.py` and
ExecuTorch runtime as the original TinyCNN example, and shows the result on
the DevKit's 480x800 MIPI DSI panel.

## What the NPU cannot do, and what it does here

The Ethos-U85 executes a Vela-compiled command stream over static shapes:
no data-dependent control flow, no scattered writes, int8/int16 only. That
rules out rasterization (edge walking, z-test, per-triangle branching). It
leaves the stages that are tensor math, and the prototype implements them as
two ExecuTorch methods in [`model/model.py`](../model/model.py):

| Method | Precision | Graph | Input | Output |
|--------|-----------|-------|-------|--------|
| `vertex` | int16 activations | `clip = bmm(pos, mvp)`, `nview = bmm(nrm, mv)`; the batch dimension is the object, each with its own matrices | 2 objects x 512 x 4 positions and normals, 2 x 4 x 4 MVP and model-view | clip-space positions, view-space normals |
| `shade` | int8 | 1x1 conv (Lambert n·L) → relu → key + ambient → `albedo * light` → depth fog → 3x3 depthwise Gaussian → clamp → 2x bilinear upscale → NCHW-to-NHWC transpose | G-buffer 3 + 3 + 1 planes, 240 x 400 | 800 x 480 x 3 interleaved RGB, the display's RGB888 layout |

The Cortex-M55 does everything in between ([`src/app_main.cpp`](../src/app_main.cpp)),
written with Helium (MVE) intrinsics:

- perspective divide and viewport, one vertex per 4-lane vector;
- an edge-function rasterizer that walks the bounding box four pixels per
  iteration: barycentric weights as affine planes stepped per row, coverage,
  the 16-bit z-test (widening loads, narrowing stores), normal interpolation
  with a vector reciprocal square root (MVE has no vector sqrt or divide, so
  it is the integer seed plus one Newton step), albedo and fog interpolation,
  and int8 quantization with `vcvtn` and narrowing predicated stores into the
  planar G-buffer;
- vector fills for the G-buffer and z-buffer clears;
- the frame copy: the ExecuTorch Ethos-U backend copies the method output
  out of the NPU scratch through a weak `arm_ethos_io_memcpy` hook; the
  runner overrides it so the 1.15 MB frame goes straight into the back frame
  buffer with 16-byte vector loads and stores, XORing 0x80 on the way (int8
  `q + 128` is the uint8 the panel wants). The output tensor is never
  written.

The frame is double-buffered: the back buffer is presented with the CDC200's
vsync-synchronised frame buffer update, and the copy into the other buffer
waits for the controller's next start-of-frame event before overwriting it
([`board/DevKit-E8/board_display.c`](../board/DevKit-E8/board_display.c)).

The scene is a checkered torus (512 vertices) and a striped sphere (480
vertices) orbiting it, 1920 triangles, animated for as long as the board
runs. Without a display (the Corstone-320 FVP target) it renders 8 frames
and prints an ASCII preview.

## Measured on the DevKit-E8

Average over 120-frame windows, Cortex-M55 HP at 400 MHz, AC6 with the app
group built `optimize: speed`, display on:

| Stage | Where | Time |
|-------|-------|------|
| Vertex transform, 2 x 512 vertices, 2 batched matmuls | NPU | 0.15 ms (0.04 ms of it copying the int16 tensors) |
| Perspective divide + viewport, 1024 vertices | CPU, Helium | 0.18 ms |
| Clear + rasterize (85 k box pixels tested, 20 k covered, ~850 front-facing triangles) | CPU, Helium | 10 ms (clear 0.3 ms) |
| Deferred shading + fog + 3x3 filter + 2x bilinear upscale, 480x800 | NPU | 8.1 ms, of which 2.5 ms is the Helium frame copy into the back buffer; vsync wait ~0 |
| Frame | | 19 ms, 50 to 53 fps |

The NPU output matches a float reference of the same shading (including
the Gaussian and the bilinear resize) within 1.6/255 on a 12 x 20 grid of
frame pixels, checked on every reported frame.

Reading the numbers:

- **The NPU stages are well under a frame.** Vertex transform is essentially
  free. Shading, filtering and upscaling a full 480x800 frame is 5.6 ms of
  NPU time; the 2.5 ms on top is the CPU moving the frame out of the NPU
  scratch, which a DMA behind the same hook would hide.
- **The round trip is memory-bound, not MAC-bound.** The shade graph has
  4.4 M MACs, which the 256-MAC NPU would finish in tens of microseconds;
  Vela reports the graph as bandwidth-limited over 96 kB and 1.15 MB tensors.
- **The Helium rasterizer is now the largest CPU item at 10 ms** for 20 k
  covered pixels, about 200 cycles per covered pixel including the ~65 %
  of bounding-box pixels that fail the coverage test. Scalar C at `-O2`
  took 15 ms for 17.6 k pixels at QVGA before the Helium rewrite.

## How the export differs from the TinyCNN example

- `model/model.py` returns a list of `MethodSpec` (module, calibration
  samples, activation bits) and `create_ai_layer.py` exports them into one
  `.pte` with two methods, using `to_edge_transform_and_lower` with a dict of
  programs and partitioners.
- Both methods take and return **quantized tensors**. The Arm backend leaves
  the float quantize/dequantize on the CPU by default; the exporter removes
  them with ExecuTorch's `quantize_input` / `quantize_output` passes
  (`executorch.exir.passes.quantize_io_pass`) and writes the resulting
  scales, zero points, shapes and C types to `ai_layer/model_io.h`, which the
  rasterizer uses to encode the G-buffer directly. No float tensor and no
  CPU operator is left in the program (`operators: []` in the layer).
- Inputs are **not memory-planned** (`MemoryPlanningPass(alloc_graph_input=False)`):
  otherwise the runtime reserves planned memory for every input and memcpys
  the caller's tensors into it on each call, 672 kB per frame for the
  G-buffer, and the shade method did not fit its pool.
- Int16 activations (`get_symmetric_a16w8_quantization_config`) work for the
  batched matmul on the U85 with Vela 5.1; the calibration samples pin the
  ranges (|pos| ≤ 1, |matrix| ≤ 4, |clip| ≤ 8) so the fixed-point scale is
  2⁻¹² everywhere.
- Vela wants 3.0 MB of scratch for the 480x800 shade graph, so the DevKit-E8
  board layer sets the ExecuTorch temp pool to 3.5 MB and the method pool
  to 1.5 MB (bulk SRAM0+SRAM1, AC6 scatter file), next to the two 1.15 MB
  frame buffers and the z-buffer. The 672 kB int8 G-buffer lives in the
  DTCM.

## Display

The board layer adds the pack's CDC200, MIPI DSI, DPHY, GPIO and ILI9806E
panel components and powers the DPHY in `main.c` as the pack's own
DevKit-e8 layer does. `board_display.c` wraps the CDC200 driver in the same
sequence the pack's VideoOut driver uses: Initialize, PowerControl, enable
the scanline-0 event, configure the display, set a frame buffer, Start.
The layer's `RTE_Device.h` (the pack's DevKit-E8 configuration) sets
RGB888, 480x800 at 60 fps, and the panel variant. The app defines
`APP_HAS_DISPLAY`, `APP_DISPLAY_WIDTH/HEIGHT` and `APP_FRAMEBUFFER_SECTION`
come from the layer, so the same `app_main.cpp` builds headless for the
FVP.

## Running it

Same three steps as the README, on the `DevKit-E8` target-type:

```bash
cbuild setup cmsis-executorch.csolution.yml --active DevKit-E8
python3 create_ai_layer.py cmsis-executorch.cbuild-mlops.yml
cbuild cmsis-executorch.csolution.yml --active DevKit-E8
```

Then **Debug** or **Run** in the CMSIS view (or ask the CMSIS Developer
Assistant to load and run it). The panel shows the animation; the console
on UART4 prints the per-stage timings every 120 frames.

Standalone (no debugger attached) the loop is paced by the panel's vsync
and reports 51 to 55 fps. If the board hard-faults straight after a flash,
see the README's troubleshooting note on the J-Link MRAM loader.

The `SSE-320-U85` FVP target still builds; its timings are not those of
hardware.
