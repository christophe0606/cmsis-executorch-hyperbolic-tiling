# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""NPU render: the tensor-shaped stages of a small 3D pipeline, for Ethos-U85.

The Ethos-U85 executes a precompiled command stream over static shapes, so it
cannot rasterize. It can, however, run the parts of a 3D pipeline that are
plain tensor math, and this module defines those as two ExecuTorch methods
that create_ai_layer.py quantizes and compiles for the NPU:

  vertex(pos, nrm, mvp, mv)  int16   Batch vertex transform: clip-space
                                     positions and view-space normals as two
                                     batched matmuls; the batch dimension is
                                     the object, each with its own matrices.
  shade(normal, albedo, depth) int8  Deferred shading of a G-buffer: Lambert
                                     lighting (1x1 conv), ambient, depth fog,
                                     a 3x3 depthwise post filter, then a 2x
                                     bilinear upscale to the panel resolution
                                     and a transpose to an interleaved
                                     RGB888 frame the display controller
                                     scans out directly.

The CPU (src/app_main.cpp) owns everything in between: perspective divide,
culling, edge-function rasterization into the G-buffer, and the z-buffer.
Shapes are fixed at export time, so the vertex batch is padded to
MAX_VERTICES, the G-buffer is FRAME_HEIGHT x FRAME_WIDTH and the output is
UPSCALE times that: the DevKit-E8 panel, 480 x 800 portrait.

The weights are hand-set (no training): a light direction, an ambient term,
a fog colour and a Gaussian kernel, so the shaded image is predictable and
the CPU-side reference in app_main.cpp can check the NPU output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

MAX_OBJECTS = 2  # bmm batch: one model-view / MVP pair per object
MAX_VERTICES = 512  # per object, padded (static shape)
FRAME_WIDTH = 240  # G-buffer, portrait: half the 480 x 800 panel in each axis
FRAME_HEIGHT = 400
UPSCALE = 2  # bilinear, on the NPU; the output is (1, H * UPSCALE, W * UPSCALE, 3)

LIGHT_DIR = (0.30, 0.50, -0.81)  # view space, towards the light, |L| = 1; the camera looks down +z
KEY_LIGHT = 0.8
AMBIENT = 0.2
FOG_COLOR = (0.10, 0.12, 0.18)
GAUSS_3X3 = [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]


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


class VertexStage(nn.Module):
    """clip = pos x mvp, nview = nrm x mv, both as batched matmuls (row-vector convention)."""

    def forward(
        self, pos: torch.Tensor, nrm: torch.Tensor, mvp: torch.Tensor, mv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip = torch.bmm(pos, mvp)  # (B, N, 4) x (B, 4, 4) -> (B, N, 4)
        nview = torch.bmm(nrm, mv)  # (B, N, 4) x (B, 4, 4) -> (B, N, 4), w = 0
        return clip, nview


class ShadeStage(nn.Module):
    """Deferred shading of a planar G-buffer, then a 3x3 Gaussian post filter."""

    def __init__(self) -> None:
        super().__init__()
        self.lambert = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.lambert.weight.data = torch.tensor(LIGHT_DIR).view(1, 3, 1, 1)
        self.post = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        kernel = torch.tensor(GAUSS_3X3)
        self.post.weight.data = (kernel / kernel.sum()).expand(3, 1, 3, 3).clone()
        self.register_buffer("fog_color", torch.tensor(FOG_COLOR).view(1, 3, 1, 1))

    def forward(self, normal: torch.Tensor, albedo: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        ndotl = torch.relu(self.lambert(normal))  # (1, 1, H, W)
        light = ndotl * KEY_LIGHT + AMBIENT
        lit = albedo * light  # broadcast over the 3 colour channels
        color = lit * (1.0 - depth) + self.fog_color * depth  # depth in [0, 1] is the fog factor
        color = torch.clamp(self.post(color), 0.0, 1.0)
        frame = nn.functional.interpolate(color, scale_factor=UPSCALE, mode="bilinear", align_corners=False)
        return frame.permute(0, 2, 3, 1)  # NCHW -> NHWC: interleaved RGB rows, the display's RGB888 layout


def _vertex_samples() -> list[tuple[torch.Tensor, ...]]:
    """Calibration that pins the int16 ranges: |pos| <= 1, |nrm| <= 1, |matrix| <= 4, |out| <= 8."""
    b, n = MAX_OBJECTS, MAX_VERTICES
    # Every tensor is a distinct object: torch.export aliases inputs that
    # share one tensor and then drops the duplicate from the graph.
    ones = lambda: torch.ones(b, n, 4)  # noqa: E731
    two = lambda: torch.full((b, 4, 4), 2.0)  # noqa: E731  1 * 2 * 4 columns = 8 at the output
    ext = torch.zeros(b, 4, 4)
    ext[:, 0, 0], ext[:, 1, 1] = 4.0, -4.0
    g = torch.Generator().manual_seed(0)
    rnd = lambda *shape: torch.rand(*shape, generator=g) * 2 - 1  # noqa: E731
    return [
        (ones(), ones(), two(), two()),
        (-ones(), -ones(), two(), two()),
        (rnd(b, n, 4), rnd(b, n, 4), ext, ext.clone()),
        (rnd(b, n, 4), rnd(b, n, 4), rnd(b, 4, 4) * 2, rnd(b, 4, 4) * 2),
    ]


def _shade_samples() -> list[tuple[torch.Tensor, ...]]:
    """Calibration that pins the int8 ranges: normal in [-1, 1], albedo and depth in [0, 1]."""
    h, w = FRAME_HEIGHT, FRAME_WIDTH
    g = torch.Generator().manual_seed(1)

    def sample(normal_value: float | None, albedo_value: float | None, depth_value: float | None):
        normal = torch.rand(1, 3, h, w, generator=g) * 2 - 1
        normal = normal / normal.norm(dim=1, keepdim=True)
        albedo = torch.rand(1, 3, h, w, generator=g)
        depth = torch.rand(1, 1, h, w, generator=g)
        if normal_value is not None:
            normal.fill_(normal_value)
        if albedo_value is not None:
            albedo.fill_(albedo_value)
        if depth_value is not None:
            depth.fill_(depth_value)
        return normal, albedo, depth

    lit = torch.tensor(LIGHT_DIR).view(1, 3, 1, 1).expand(1, 3, h, w).clone()
    return [
        (lit, torch.ones(1, 3, h, w), torch.zeros(1, 1, h, w)),  # brightest: n = L, white, no fog
        sample(-1.0, 0.0, 1.0),  # darkest / full fog
        sample(None, None, None),
        sample(None, None, None),
    ]


def get_methods() -> list[MethodSpec]:
    torch.manual_seed(0)
    return [
        MethodSpec("vertex", VertexStage().eval(), _vertex_samples(), activation_bits=16),
        MethodSpec("shade", ShadeStage().eval(), _shade_samples(), activation_bits=8),
    ]
