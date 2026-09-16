# Helium resolution, antialiasing and reflection budget

The firmware defaults to **480x800, AA none, at most 12 rounds**.
All Möbius transforms, reflections, texture gathers, color composition and AA
run on M55/Helium. Full resolution writes directly to RGB888. Half resolution
uses the Ethos `upscale` method to enlarge the already antialiased colors.
The generated AI layer is included: ordinary VS Code Build uses this renderer.

UART4, 115200 baud:

```text
scale full          # 480x800 geometry; native display resolution
scale half          # 240x400 geometry, Ethos bilinear 2x enlargement
aa full             # four samples at (+/-0.25, +/-0.25) geometry pixels everywhere
aa partial          # 2x2 samples in rectangular boundary bands only
aa none             # one center sample everywhere
iterations 12       # maximum rounds, 1..40; three ordered mirrors per round
iterations 40       # reference detail budget
animation off       # reproducible geometry and texture for comparisons
status
reset               # restore full resolution, AA none, 12 rounds, animation on
```

The existing symmetry, geometry, colors, texture zoom, preview and probe commands
remain available. Commands received during a frame are applied between frames.
The A/B convention remains the same as the main-branch Helium renderer.

## How rendering works

### Partial AA

MCP uses `antialiasing(mode="none"|"partial"|"full")`. The previous
`on=true/false` argument remains accepted as a compatibility alias for full/none;
console `aa on/off` also remains accepted. Status always reports the mode name.
When the COM connection closes and reopens after a firmware update, the UART
bridge rediscovers tool schemas and notifies connected MCP clients to refresh
their tool list. If the update leaves the COM connection healthy, restart the
bridge and reconnect clients to refresh the schemas.

Partial mode estimates the projected triangle area using the hyperbolic area
`A = pi * (1 - 1/p - 1/q - 1/r)` and the projection's local metric. The cutoff
is **3 geometry pixels squared**, measured before half-resolution upscaling.
For disk geometry at render width `W`, estimated area is
`A * W^2 * (1-r^2)^2 / 16`. The center square is inscribed in the circle where
that estimate reaches 3 pixels squared. All pixels outside the square receive
2x2 AA, including the entire fine-triangle annulus and additional regions by
the square's sides. The original outside-disk background shortcut still applies.
This is a local size estimate, not an exact measurement of finite triangles.

For portrait plane geometry, the metric gives area
`A * W^2 * cos(pi*screen_x/2)^2 / pi^2`. Its boundaries are the left/right
sides, so only two vertical bands need AA. The same 3-pixel cutoff determines
their width. Animation is a hyperbolic isometry and does not require moving
these projection-based regions every frame.

The center is rounded inward to four-pixel vector boundaries. Rectangles are
computed when geometry, resolution or symmetry changes, then shared by HP and
HE. Each strip is partitioned into disjoint rectangles: the center uses the
existing four-pixels-per-vector kernel, while the bands use the existing
four-samples-per-pixel kernel. There are no per-pixel circle tests for selecting
AA, duplicate center rendering, or additional frame buffers. Halo rows use the
same global coverage as adjacent strips before Ethos interpolation.

The cutoff is `kPartialTrianglePixels` in `src/tiling_antialiasing.hpp` for later
visual tuning. The timings below predate partial mode and do not measure it.

### Sampling

Each subpixel independently goes through Möbius, reflection, edge classification,
texture lookup and shading. A uint16 buffer sums **final colors**, then divides
by four with rounding. Coordinates, parity and edge masks are never averaged.
RGB bytes are rounded once for each sample, so accumulation cannot overflow.

The 16x240 strip buffers fit either resolution without allocating a full-screen
G-buffer. Half-resolution strips advance by 14 rows and have one halo row on
either side; the halo is discarded after enlargement to avoid horizontal seams.
Full resolution uses two strips across and no interpolation. Separable mapping
tables avoid repeated exponentials and trig functions for plane geometry.
Vectors entirely outside the disk write the background without reflecting.

## Choosing a limit

The mathematical boundary is infinitely far away, but a finite-resolution
display cannot show arbitrarily small triangles. The limit is a visual detail
budget, not an exact convergence requirement. Judge whether the large tiles
and alternating colors remain intact, and whether any differences are limited
to one- or two-pixel details near the rim. A reflection count alone does not
measure projected triangle area; this implementation uses a tunable cap,
not an automatic triangle-area stopping test.

Helium already exits when no lane in a vector reflects during a round. Lowering
40 to 12 therefore saves the long tail, rather than reducing total work by
40/12. The UART reports average rounds per vector and how many vectors reach
the cap. The latter is conservative: a vector may have converged on its final
allowed round. Neither statistic is a perceptual pass/fail gate.

Full-resolution four-sample host comparisons against 40 rounds, using float64
geometry, three symmetry presets and static/animated views:

| Maximum rounds | Mean absolute RGB difference, across tested views (0..255) |
|---|---:|
| 4 | 2.81–12.12 |
| 6 | 0.28–3.90 |
| 8 | 0.03–1.05 |
| 12 | 0.0001–0.103 |

Four to six rounds visibly lose detail in the difficult (2,4,5) preset. Twelve
is a conservative starting point; differences from 40 in these views are tiny
rim details. Eight is available as a more aggressive speed/detail tradeoff.
These host numbers do not include Helium float32 roundoff near the boundary.

## Board results

Measured on DevKit-E8 M55_HP at 400 MHz, using ULINKplus, with static disk
geometry and the default (2,4,5) preset:

| Setting | Render time | Approximate render rate |
|---|---:|---:|
| Full, 2x2 AA, 40 rounds | 378.9 ms | 2.64 fps |
| Full, 2x2 AA, 12 rounds | 374.0 ms | 2.67 fps |
| Full, 2x2 AA, 8 rounds | 366.0 ms | 2.73 fps |
| Half, 2x2 AA, 12 rounds, Ethos upscale | 120.8 ms | 8.28 fps |
| Half, no AA, 12 rounds, Ethos upscale | 39.7 ms | 25.19 fps |
| Full, no AA, 12 rounds | 97.5 ms | 10.26 fps |

Times sum per-strip DWT durations, including geometry, final-color averaging,
framebuffer writes, and NPU execution/copies where applicable, plus texture
update. They exclude console output and display synchronization. Interrupted
frames are excluded. Animation and plane geometry can change timings.
Firmware prints timing periodically rather than on every frame to limit UART overhead.
Skipping all-background vectors reduced full-AA/40 static rendering from
526.3 to 378.9 ms without changing those samples' colors.

The full-resolution 12/40 comparison captured from the board has mean absolute
RGB difference **0.091/255**. All 1,307 pixels whose maximum channel difference
exceeds 16/255 lie within two screen pixels of the disk rim in this static
view. The main tiles agree visually. At eight rounds, 4,031 such pixels lie
farther inward, with only an 8 ms saving over twelve rounds. This supports
**12 rounds** as the default quality choice, and **half resolution with AA**
when smoother animation matters more than native resolution.

`out/helium-12-vs-40.png` shows 40 rounds / 12 rounds / difference x4.
`out/helium-board-iterations.png` shows 8 / 12 / 40 rounds. These are actual
framebuffer captures; they are distinct from the float64 host comparisons.

Reproduce the comparison and the spatial contract tests using the project venv:

```powershell
uv run --python .venv/Scripts/python.exe python model/evaluate_iterations.py
uv run --python .venv/Scripts/python.exe python -m unittest discover -s model -p test_render.py -v
```

`out/helium-iterations/` contains PNGs and `report.json`. Comparison images are
4 / 6 / 8 / 12 / 40 rounds, left to right. Tests verify AA arithmetic, distinct
subpixels, mirror geometry, and strip equivalence to whole-image interpolation.

The board setup uses **ULINKplus only**. The solution records ULINKplus@pyOCD;
an enumerated J-Link entry must not be used as a fallback for this setup.

## Helium optimization

AA uses one vector per pixel, with lanes representing the four positions in
the 2x2 grid. Colors are shaded independently and reduced with the same rounded
integer average. This avoids four passes through the strip and repeated
accumulator loads. The no-AA specialization still handles four pixels per vector.

Mapping feeds the kernel directly from the separable tables. Removing the two
coordinate strips saves 30,720 bytes of DTCM and about 7.1 MB of coordinate
write/read traffic per half-resolution AA frame, including halo rows.

For all three supported presets, mirror 1 is `x = abs(x)` and mirror 3 sorts
`(x, y)` because its normal is proportional to `(-1, 1, 0)`. Only mirror 2
needs a general Minkowski reflection. A new preset must preserve those mirror
orientations or provide its own implementation. Parity is tracked by predicate
XOR, with an inline assembly register constraint preventing AC6 from retaining
multiple predicates and spilling them to the stack.

The kernel keeps homogeneous coordinates `(2x, 2y, 1+r²)` and `w=1-r²`,
removing the initial reciprocal. The final projection divides by `z+w`, and
the squared edge threshold is multiplied by `w²`. Host tests compare folding,
parity, projection and edge distances with the general normalized formulation.
Float32 rounding can still differ at fine tile boundaries; board captures are
the visual acceptance check.

The upscaler method and input binding are prepared before the frame loop.
Repeated execution uses the existing method and temporary arenas directly,
without allocating input/output `std::vector` wrappers per strip. Initialization
still uses the module's standard C++ ownership objects. The hot kernel remains
in MRAM; moving it to ITCM was not retained because the C++ unwind-table
relocation could not span the two regions with the current link configuration.

### Optimized board measurements (2026-09-14)

The current build compiles and passes all five host tests. Measurements after
a board power cycle, using CMSIS Developer Assistant and ULINKplus:

| Setting | Geometry | Upscale | Total render |
|---|---:|---:|---:|
| Half, AA, 12 rounds, static default disk | 80.3 ms | 7.5 ms | 90.9 ms |
| Same setting, intermittent slower upscale | 80.3–80.8 ms | 22.2–22.4 ms | 105.6–106.2 ms |
| Half, AA, 12 rounds, sampled animated views | 92.2–99.9 ms | 7.5 ms | 102.9–110.5 ms |
| Full, AA, 12 rounds, sampled animated views | 321.1–336.7 ms | — | 325.4–340.9 ms |

The normal static half-resolution frame is 24.8% shorter than the original
120.8 ms frame (1.33x render rate). These are per-frame observations, not
percentiles from a continuous profiler. Intermittent upscaler delays remain
unexplained. **The 70 ms target is not reached.**

The actual static framebuffer comparison against `helium-half-aa-i12.rgb`
has mean RGB difference **0.13557/255**; 368,602 of 384,000 pixels (96.0%)
are identical. 1,590 pixels exceed 16/255 maximum channel difference, of
which 456 are more than four screen pixels inside the rim. Visual inspection
shows scattered fine differences, mostly around the rim, without large
same-color patches. This comparison covers the static default disk preset;
it does not establish visual equivalence for every animated or plane view.
`out/helium-optimized-comparison.png` shows original / optimized / difference
x4; `out/helium-optimized-comparison.json` records the metrics.

**Stability validation remains open.** During the longer run, the debugger
reported `Fault_Handler` with `app_main`'s display presentation on the stack,
then became unresponsive before fault registers could be read. The MCP restart
also timed out. The fault cause has not been established, so these performance
and image results must not be interpreted as a successful stability test.
Earlier repeated debug resets also caused secure-enclave synchronization and
display initialization failures. A power cycle recovered startup; it did not
resolve the later fault investigation.
