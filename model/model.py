# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Hyperbolic tiling: the tensor-shaped half of a full-screen fragment shader, for Ethos-U85.

A port of christophe0606/shader_linux_glsl, a GLSL fragment shader that tiles
the Poincare disk with reflections of a camera image, split the way the NPU
render demo splits a 3D pipeline:

  CPU (Helium)  the branchy per-pixel geometry: Moebius animation, the
                iterated hyperbolic reflections with early exit, the edge
                distance test, tile parity and the texture coordinate of
                every pixel. It writes a "tiling G-buffer" at TILE_HEIGHT x
                TILE_WIDTH: a texel index, a parity mask, an edge mask and
                an inside-the-disk mask.
  NPU           tile(...): the texture lookup as a gather (TOSA GATHER on the
                U85), the tile / edge / background colouring as masked
                blends, then a 2x bilinear upscale to the 480x800 panel and
                the transpose to interleaved RGB888.

The colours are method inputs, so the console commands that stand in for the
original demo's MCP tools change them without a re-export.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

TILE_WIDTH = 240  # geometry resolution, portrait: half the 480 x 800 panel in each axis
TILE_HEIGHT = 400
UPSCALE = 2  # bilinear, on the NPU
TEXTURE_SIZE = 128  # the (camera) texture is TEXTURE_SIZE x TEXTURE_SIZE RGB


@dataclass
class MethodSpec:
    """One ExecuTorch method: the module, its calibration/example inputs and the activation width."""

    name: str
    module: nn.Module
    samples: list[tuple[torch.Tensor, ...]] = field(default_factory=list)
    activation_bits: int = 8  # 8 or 16 (int16 activations, int8 weights)

    @property
    def example(self) -> tuple[torch.Tensor, ...]:
        return self.samples[0]


class TileStage(nn.Module):
    """Compose the frame from the tiling G-buffer and the texture.

    texture   (T*T, 3)      RGB texels in [0, 1], row-major
    index     (H*W,) int32  texel index of every pixel (the CPU's texture coordinate)
    parity    (1, 1, H, W)  1 for tile A, 0 for tile B
    edge      (1, 1, H, W)  1 on a tile edge
    inside    (1, 1, H, W)  1 inside the disk, 0 for the background
    tile_a, tile_b, edge_color, background   (1, 3, 1, 1) colours in [0, 1]

    Returns (1, H * UPSCALE, W * UPSCALE, 3) in [0, 1].
    """

    def forward(
        self,
        texture: torch.Tensor,
        index: torch.Tensor,
        parity: torch.Tensor,
        edge: torch.Tensor,
        inside: torch.Tensor,
        tile_a: torch.Tensor,
        tile_b: torch.Tensor,
        edge_color: torch.Tensor,
        background: torch.Tensor,
    ) -> torch.Tensor:
        texel = torch.index_select(texture, 0, index)  # (H*W, 3): the texture lookup
        texel = texel.reshape(1, TILE_HEIGHT, TILE_WIDTH, 3).permute(0, 3, 1, 2)  # (1, 3, H, W)
        tile = tile_a * parity + tile_b * (1.0 - parity)  # the tile's own colour
        color = 0.5 * texel + 0.5 * tile  # as the shader: half texture, half tile colour
        color = edge_color * edge + color * (1.0 - edge)
        color = color * inside + background * (1.0 - inside)
        frame = nn.functional.interpolate(color, scale_factor=UPSCALE, mode="bilinear", align_corners=False)
        return frame.permute(0, 2, 3, 1)  # NCHW -> NHWC: interleaved RGB rows, the display's RGB888 layout


def _tile_samples() -> list[tuple[torch.Tensor, ...]]:
    """Calibration that pins every range to [0, 1]; the index is int32 and not quantized."""
    h, w, t = TILE_HEIGHT, TILE_WIDTH, TEXTURE_SIZE
    g = torch.Generator().manual_seed(2)

    def sample(fill: float | None):
        texture = torch.rand(t * t, 3, generator=g)
        index = torch.randint(0, t * t, (h * w,), generator=g, dtype=torch.int32)
        masks = [torch.randint(0, 2, (1, 1, h, w), generator=g).float() for _ in range(3)]
        colors = [torch.rand(1, 3, 1, 1, generator=g) for _ in range(4)]
        if fill is not None:
            texture.fill_(fill)
            for m in masks:
                m.fill_(fill)
            for c in colors:
                c.fill_(fill)
        return (texture, index, *masks, *colors)

    return [sample(1.0), sample(0.0), sample(None), sample(None)]


def get_methods() -> list[MethodSpec]:
    torch.manual_seed(0)
    return [MethodSpec("tile", TileStage().eval(), _tile_samples(), activation_bits=8)]
