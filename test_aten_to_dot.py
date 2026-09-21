# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Topology and Graphviz syntax checks for the ATen visualizer."""

import re
import shutil
import subprocess
import unittest

import torch

from aten_to_dot import program_to_dot


class BranchedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("offset", torch.ones(2))

    def forward(self, x, y):
        squared = x * x
        merged = torch.cat([squared, y + self.weight + self.offset])
        return {"merged": merged, "nested": (squared, squared, 7)}


class DotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = torch.export.export(BranchedModel(), (torch.ones(2), torch.zeros(2)))
        cls.dot = program_to_dot(cls.program, 'Graph "quoted" \\ path\nnext line')

    def test_input_roles_and_metadata(self):
        self.assertIn('User Input\\nx\\nfloat32 [2]', self.dot)
        self.assertIn('Parameter\\np_weight\\nweight', self.dot)
        self.assertIn('Buffer\\nb_offset\\noffset', self.dot)
        self.assertIn('aten.mul.Tensor', self.dot)

    def test_repeated_and_nested_operands(self):
        nodes = list(self.program.graph.nodes)
        ids = {n: f"n{i}" for i, n in enumerate(nodes)}
        x = next(n for n in nodes if n.name == "x")
        mul = next(n for n in nodes if n.target == torch.ops.aten.mul.Tensor)
        cat = next(n for n in nodes if n.target == torch.ops.aten.cat.default)
        self.assertIn(f'{ids[x]} -> {ids[mul]} [label="args[0]"]', self.dot)
        self.assertIn(f'{ids[x]} -> {ids[mul]} [label="args[1]"]', self.dot)
        self.assertIn(f'{ids[mul]} -> {ids[cat]} [label="args[0][0]"]', self.dot)
        self.assertIn(f'{ids[mul]} -> out1 ', self.dot)
        self.assertIn(f'{ids[mul]} -> out2 ', self.dot)

    def test_all_outputs_including_constants_and_no_dangling_edges(self):
        self.assertIn('User Output 3\\n7', self.dot)
        self.assertEqual(len(re.findall(r'^  out\d+ \[', self.dot, re.M)), 4)
        declared = set(re.findall(r'^  (\w+) \[', self.dot, re.M))
        for source, destination in re.findall(r'^  (\w+) -> (\w+)', self.dot, re.M):
            self.assertIn(source, declared)
            self.assertIn(destination, declared)

    def test_layout_validation(self):
        self.assertIn('rankdir=TB', program_to_dot(self.program, rankdir="TB"))
        with self.assertRaises(ValueError):
            program_to_dot(self.program, rankdir='LR; invalid')

    @unittest.skipUnless(shutil.which("dot"), "Graphviz is not installed")
    def test_graphviz_accepts_escaped_labels_and_renders_topology(self):
        rendered = subprocess.run(
            ["dot", "-Tsvg"], input=self.dot, text=True,
            encoding="utf-8", capture_output=True, check=True,
        )
        self.assertIn('<svg', rendered.stdout)
        self.assertIn('aten.mul.Tensor', rendered.stdout)
        self.assertIn('class="edge"', rendered.stdout)


if __name__ == "__main__":
    unittest.main()
