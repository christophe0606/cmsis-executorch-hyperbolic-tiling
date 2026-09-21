#!/usr/bin/env python3
# Copyright 2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert a torch.export ATen program (.pt2) to Graphviz DOT.

    python aten_to_dot.py model.pt2 -o model.dot
    dot -Tsvg model.dot -o model.svg

program_to_dot() also accepts an in-memory ExportedProgram. No Graphviz Python
package is needed. Each ATen operation is a node; parameters and buffers are
shown separately from user inputs. Rendering requires Graphviz's dot command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _quote(text: str) -> str:
    """Escape plain DOT strings (not HTML or record labels)."""
    return json.dumps(text, ensure_ascii=False)


def _leaves(value, path: str):
    """Keep argument positions, including repeated and nested dependencies."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def _short(value, limit: int = 100) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tensor_info(value) -> str:
    import torch

    if isinstance(value, torch.Tensor):
        shape = ", ".join(str(d) for d in value.shape)
        return f"{str(value.dtype).removeprefix('torch.')} [{shape}]"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_tensor_info(v) for v in value) + ")"
    return "" if value is None else _short(value)


def program_to_dot(
    program, name: str = "ATen graph", rankdir: str = "LR", *, compact: bool = False
) -> str:
    """Describe an ExportedProgram's complete dataflow, without modifying it.

    Output nodes follow the flattened export signature, so tuple/dict returns,
    repeated results, constant returns and mutation outputs remain explicit.
    Edge labels identify operand positions; node labels include tensor metadata
    and scalar arguments. Module paths are included when export provides them.
    Compact labels show only operator names, input/parameter identifiers and
    output positions, without tensor metadata or scalar arguments.
    """
    from torch.fx import Node

    if rankdir not in {"LR", "TB"}:
        raise ValueError("rankdir must be LR or TB")
    nodes = list(program.graph.nodes)
    ids = {node: f"n{index}" for index, node in enumerate(nodes)}
    inputs = {
        spec.arg.name: spec
        for spec in program.graph_signature.input_specs
        if hasattr(spec.arg, "name")
    }
    lines = [
        "digraph aten {",
        f"  graph [rankdir={rankdir}, label={_quote(name)}, labelloc=t, fontname=Helvetica];",
        '  node [shape=box, style="rounded,filled", fillcolor="#eef2f6", fontname=Helvetica, fontsize=10];',
        '  edge [color="#64748b", fontname=Helvetica, fontsize=8];',
    ]

    def add_node(node_id, label, color, shape="box"):
        lines.append(
            f"  {node_id} [label={_quote(label)}, fillcolor={_quote(color)}, shape={shape}];"
        )

    def connect(source, destination, label):
        lines.append(f"  {ids[source]} -> {destination} [label={_quote(label)}];")

    for node in nodes:
        if node.op == "output":
            specs = program.graph_signature.output_specs
            for index, (_, value) in enumerate(_leaves(node.args[0], "output")):
                spec = specs[index] if index < len(specs) else None
                kind = spec.kind.name.replace("_", " ").title() if spec else "Output"
                label = f"{kind} {index}"
                if not compact and spec and spec.target is not None:
                    label += f"\n{spec.target}"
                if not compact:
                    label += f"\n{value.name if isinstance(value, Node) else repr(value)}"
                if not compact and isinstance(value, Node):
                    info = _tensor_info(value.meta.get("val"))
                    if info:
                        label += f"\n{info}"
                output_id = f"out{index}"
                add_node(output_id, label, "#dcfce7", "oval")
                if isinstance(value, Node):
                    connect(value, output_id, "result")
            continue

        color, shape = "#eef2f6", "box"
        if node.op == "placeholder":
            spec = inputs.get(node.name)
            kind = spec.kind.name if spec else "USER_INPUT"
            title = kind.replace("_", " ").title()
            label = f"{title}\n{node.name}"
            if spec and spec.target is not None:
                label += f"\n{spec.target}"
            if compact:
                label = str(spec.target) if spec and spec.target is not None else node.name
            color = "#dbeafe" if kind == "USER_INPUT" else "#fef3c7"
            shape = "oval" if kind == "USER_INPUT" else "box"
        elif compact:
            label = str(node.target)
            if node.op == "get_attr":
                color = "#fef3c7"
        else:
            label = f"{node.name}\n{node.target}"
            if node.op == "get_attr":
                label = f"Attribute\n{node.target}"
                color = "#fef3c7"
            stack = node.meta.get("nn_module_stack", {})
            if stack:
                module_path, _ = list(stack.values())[-1]
                if module_path:
                    label += f"\nmodule: {module_path}"
            constants = [
                f"{path}={_short(repr(value))}"
                for path, value in [*_leaves(node.args, "args"), *_leaves(node.kwargs, "kwargs")]
                if not isinstance(value, Node)
            ]
            if constants:
                label += "\n" + "\n".join(constants)
        info = "" if compact else _tensor_info(node.meta.get("val"))
        if info:
            label += f"\n{info}"
        add_node(ids[node], label, color, shape)
        for path, value in [*_leaves(node.args, "args"), *_leaves(node.kwargs, "kwargs")]:
            if isinstance(value, Node):
                connect(value, ids[node], path)

    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ATen .pt2 file saved by torch.export.save")
    parser.add_argument("-o", "--output", type=Path, help="output DOT file (default: input with .dot extension)")
    parser.add_argument("--rankdir", choices=("LR", "TB"), default="LR", help="left-to-right or top-to-bottom layout")
    parser.add_argument("--compact", action="store_true", help="show only operator names on operation nodes; omit tensor metadata and scalar arguments")
    args = parser.parse_args()

    from create_ai_layer import run_in_venv

    run_in_venv()
    import torch

    output = args.output or args.input.with_suffix(".dot")
    if output.resolve() == args.input.resolve():
        parser.error("output must differ from the input file")
    program = torch.export.load(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        program_to_dot(program, args.input.stem, args.rankdir, compact=args.compact),
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
