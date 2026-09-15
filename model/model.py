# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Helium shades and averages subpixel colors; Ethos enlarges half-scale RGB.

The 16x240 RGB block includes halo rows to join enlarged strips without seams.
Full-resolution firmware bypasses this method entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

TILE_WIDTH = 240  # strip width; panel width at half resolution
TILE_HEIGHT = 16  # overlapped strips; firmware averages AA before enlargement
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


class UpscaleStage(nn.Module):
    """Bilinear 2x enlargement from RGB planes to interleaved display bytes."""

    def forward(
        self,
        texel: torch.Tensor,
    ) -> torch.Tensor:
        frame = nn.functional.interpolate(
            texel, scale_factor=UPSCALE, mode="bilinear", align_corners=False
        )
        return frame.permute(
            0, 2, 3, 1
        )  # NCHW -> NHWC: interleaved RGB rows, the display's RGB888 layout


def _tile_samples() -> list[tuple[torch.Tensor, ...]]:
    """Calibration that pins every range to [0, 1]."""
    h, w = TILE_HEIGHT, TILE_WIDTH
    g = torch.Generator().manual_seed(2)

    def sample(fill: float | None):
        texel = torch.rand(1, 3, h, w, generator=g)
        if fill is not None:
            texel.fill_(fill)
        return (texel,)

    return [sample(1.0), sample(0.0), sample(None), sample(None)]


def get_methods() -> list[MethodSpec]:
    torch.manual_seed(0)
    return [
        MethodSpec("upscale", UpscaleStage().eval(), _tile_samples(), activation_bits=8)
    ]
