# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Spatial contracts for firmware AA and overlapping half-scale strips."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from evaluate_iterations import normals

from model import UpscaleStage


class RenderTests(unittest.TestCase):
    def test_specialized_mirrors_and_predicate_parity(self):
        # Independently compare the specialized fold with the general formula.
        # Use float64 here to test the algebra; board captures check float32 art.
        rng = np.random.default_rng(2026)
        xy = rng.uniform(-0.65, 0.65, (4096, 2))
        radius = np.sum(xy * xy, axis=1)
        initial = np.column_stack((2 * xy, 1 + radius)) / (1 - radius[:, None])
        for preset in range(3):
            ns = normals(preset)
            self.assertGreater(ns[0, 0], 0)
            np.testing.assert_array_equal(ns[0, 1:], 0)
            self.assertEqual(ns[2, 2], 0)
            np.testing.assert_allclose(ns[2], [-2**-0.5, 2**-0.5, 0], atol=1e-14)
            w = 1 - radius
            general, fast = initial.copy(), initial * w[:, None]
            counts = np.zeros(len(xy), dtype=np.int32)
            parity = np.zeros(len(xy), dtype=bool)
            for _ in range(12):
                for i, n in enumerate(ns):
                    d = general @ (n * [1, 1, -1])
                    counts += d < 0
                    general -= 2 * np.minimum(d, 0)[:, None] * n
                    if i == 0:
                        parity ^= fast[:, 0] < 0
                        fast[:, 0] = np.abs(fast[:, 0])
                    elif i == 2:
                        parity ^= fast[:, 1] < fast[:, 0]
                        fast[:, :2] = np.sort(fast[:, :2], axis=1)
                    else:
                        d = fast @ (n * [1, 1, -1])
                        parity ^= d < 0
                        fast -= 2 * np.minimum(d, 0)[:, None] * n
            np.testing.assert_allclose(fast / w[:, None], general, atol=1e-10, rtol=1e-10)
            np.testing.assert_array_equal(parity, counts % 2 != 0)
            disk = general[:, :2] / (general[:, 2:] + 1)
            projected = fast[:, :2] / (fast[:, 2:] + w[:, None])
            np.testing.assert_allclose(projected, disk, atol=1e-10, rtol=1e-10)
            for n in ns:
                d = general @ (n * [1, 1, -1])
                scaled_d = fast @ (n * [1, 1, -1])
                np.testing.assert_allclose(scaled_d**2 / w**2, d**2, atol=1e-10, rtol=1e-10)

    def test_halo_after_aa_matches_whole_image(self):
        torch.manual_seed(1)
        for height in (28, 31, 400):
            samples = torch.randint(0, 256, (4, 3, height, 240)).float()
            # Average four independently shaded colors before bilinear upscale.
            average = samples.mean(0, keepdim=True)
            expected = F.interpolate(
                average, scale_factor=2, mode="bilinear", align_corners=False
            )
            got = torch.empty_like(expected)
            for by in range(0, height, 14):
                rows = torch.arange(by - 1, by + 15).clamp(0, height - 1)
                block = UpscaleStage()(average[:, :, rows, :]).permute(0, 3, 1, 2)
                count = min(14, height - by) * 2
                got[:, :, by * 2 : by * 2 + count] = block[:, :, 2 : 2 + count]
            torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_four_offsets_are_distinct_and_centered(self):
        offsets = np.array([(x, y) for y in (-0.25, 0.25) for x in (-0.25, 0.25)])
        np.testing.assert_array_equal(offsets.mean(0), [0, 0])
        self.assertEqual(len(np.unique(offsets, axis=0)), 4)

    def test_integer_color_average_does_not_overflow(self):
        for value in (0, 1, 127, 128, 254, 255):
            acc = np.zeros(4, dtype=np.uint16)
            for _ in range(4):
                acc += value
            np.testing.assert_array_equal((acc + 2) // 4, value)

    def test_reference_mirrors_preserve_minkowski_norm(self):
        point = np.array([0.3, -0.2, np.sqrt(1 + 0.3**2 + 0.2**2)])
        for preset in range(3):
            for normal in normals(preset):
                metric = np.array([1, 1, -1])
                reflected = point - 2 * np.sum(point * normal * metric) * normal
                self.assertAlmostEqual(np.sum(reflected**2 * metric), -1, places=12)


if __name__ == "__main__":
    unittest.main()
