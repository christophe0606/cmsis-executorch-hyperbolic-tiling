# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Compare finite reflection budgets against 40 rounds at display resolution.

Float64 host reference, using the firmware's texture and A/B convention.
Board captures remain the check for Helium float32 and display output.
"""

import argparse
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np


def save_png(path, rgb):
    def chunk(kind, data):
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data))
        )

    h, w, _ = rgb.shape
    data = b"".join(b"\0" + row.tobytes() for row in rgb.astype(np.uint8))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(data))
        + chunk(b"IEND", b"")
    )


def normals(preset):
    p, q, r = ((2, 4, 5), (2, 4, 7), (4, 4, 4))[preset]
    alpha, beta, gamma = (math.pi / n for n in (p, q, r))
    a = math.acosh(
        (math.cos(gamma) * math.cos(beta) + math.cos(alpha))
        / (math.sin(gamma) * math.sin(beta))
    )
    b = math.acosh(
        (math.cos(gamma) * math.cos(alpha) + math.cos(beta))
        / (math.sin(gamma) * math.sin(alpha))
    )
    c = math.acosh(
        (math.cos(alpha) * math.cos(beta) + math.cos(gamma))
        / (math.sin(alpha) * math.sin(beta))
    )
    u = math.cosh(c) / math.tanh(a) - math.cosh(b) / math.sinh(a)
    v = math.cosh(c)
    points = np.array(
        [
            [0, 0, 1],
            [0, math.sinh(a), math.cosh(a)],
            [math.sqrt(v * v - u * u - 1), u, v],
        ]
    )
    ns = -np.cross(points, np.roll(points, -1, axis=0)) * np.array([1, 1, -1])
    return ns / np.sqrt(ns[:, 0] ** 2 + ns[:, 1] ** 2 - ns[:, 2] ** 2)[:, None]


def texture():
    y, x = np.mgrid[:128, :128]
    u, v = (x + 0.5) / 128, (y + 0.5) / 128
    tex = np.stack(
        [
            0.5 + 0.5 * np.sin(6.2832 * u),
            0.5 + 0.5 * np.sin(6.2832 * v + 2),
            0.5 + 0.5 * np.sin(6.2832 * (u + v) * 0.5),
        ],
        axis=-1,
    )
    distance = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
    tex[(distance > 0.16) & (distance < 0.22)] = [1, 1, 0.9]
    tex[((x // 16 + y // 16) % 2 == 0) & (u > 0.7) & (v > 0.7)] *= 0.3
    return np.floor(tex * 255 + 0.5) / 255


def render_budgets(width, height, preset, time, budgets=(4, 6, 8, 12, 40), aa=True):
    ns, tex = normals(preset), texture()
    images = {n: np.zeros((height, width, 3)) for n in budgets}
    offsets = (
        ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))
        if aa
        else ((0, 0),)
    )
    for ox, oy in offsets:
        iy, ix = np.mgrid[:height, :width]
        x, y = (
            (ix + 0.5 + ox - width / 2) * 2 / width,
            (height / 2 - iy - 0.5 - oy) * 2 / width,
        )
        if time is not None:
            z = (x + 1j * y - 0.5 * np.cos(2 * np.pi * time * 0.1)) / (
                1 - 0.5 * np.cos(2 * np.pi * time * 0.1) * (x + 1j * y)
            )
            z *= np.exp(1j * np.deg2rad(time * 5))
            x, y = z.real, z.imag
        radius = x * x + y * y
        inside = radius < 1
        x, y, radius = (np.where(inside, a, 0) for a in (x, y, radius))
        hx, hy, hz = (
            2 * x / (1 - radius),
            2 * y / (1 - radius),
            (1 + radius) / (1 - radius),
        )
        parity = np.zeros_like(inside)
        for n in range(1, max(budgets) + 1):
            for nx, ny, nz in ns:
                distance = hx * nx + hy * ny - hz * nz
                parity ^= distance < 0
                step = 2 * np.minimum(distance, 0)
                hx -= step * nx
                hy -= step * ny
                hz -= step * nz
            if n in budgets:
                edge = np.zeros_like(inside)
                for nx, ny, nz in ns:
                    edge |= (hx * nx + hy * ny - hz * nz) ** 2 <= math.cosh(0.01) - 1
                zoom = (1, 0.5, 0.5)[preset]
                u = np.mod(zoom * 4 * (0.5 - hx / (hz + 1)) + 0.6, 1)
                v = np.mod(zoom * 4 * (0.5 - hy / (hz + 1)) + 0.5, 1)
                color = (
                    tex[
                        (v * 128).astype(int).clip(0, 127),
                        (u * 128).astype(int).clip(0, 127),
                    ]
                    * 0.5
                )
                color += (
                    np.stack([parity, np.zeros_like(parity), ~parity], axis=-1) * 0.5
                )
                color[edge | ~inside] = 0
                images[n] += color / len(offsets)
    return {n: np.floor(im * 255 + 0.5).astype(np.uint8) for n, im in images.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/helium-iterations"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for preset, time in ((0, None), (0, 1.7), (1, 7.3), (2, 12.1)):
        images = render_budgets(args.width, args.height, preset, time)
        name = f"p{preset}-t{time}"
        for n, im in images.items():
            error = np.abs(im.astype(np.int16) - images[40].astype(np.int16))
            reports.append(
                {
                    "preset": preset,
                    "time": time,
                    "iterations": n,
                    "mean_error": float(error.mean()),
                    "pixels_over_16": int((error.max(axis=-1) > 16).sum()),
                }
            )
            save_png(args.output / f"{name}-i{n}.png", im)
        save_png(
            args.output / f"{name}-comparison.png",
            np.concatenate([images[n] for n in (4, 6, 8, 12, 40)], axis=1),
        )
        print(
            name,
            [r for r in reports if r["preset"] == preset and r["time"] == time],
            flush=True,
        )
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
