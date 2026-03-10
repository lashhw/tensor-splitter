#!/usr/bin/env python3
"""Reorder split-entry slice ops in TFLite models."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubgraphSummary:
    moved_ops: int
    moved_groups: int
    operator_count: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tflite", help="Input .tflite model.")
    parser.add_argument("output_tflite", help="Output .tflite model.")
    return parser.parse_args()


def is_slice_operator(model, operator) -> bool:
    code = model["operator_codes"][operator.get("opcode_index", 0)]
    return code.get("builtin_code") in {"SLICE", "STRIDED_SLICE"}


def reorder_subgraph(model, subgraph):
    operators = subgraph.get("operators", [])
    producer_by_tensor = {}
    consumers_by_source_tensor = defaultdict(list)

    for operator_index, operator in enumerate(operators):
        for tensor_index in operator.get("outputs", []):
            if tensor_index >= 0:
                producer_by_tensor[tensor_index] = operator_index

        inputs = operator.get("inputs", [])
        if inputs and inputs[0] >= 0:
            consumers_by_source_tensor[inputs[0]].append(operator_index)

    groups = []
    moved_indices = set()

    for source_tensor, consumer_indices in consumers_by_source_tensor.items():
        if len(consumer_indices) < 2:
            continue

        producer_index = producer_by_tensor.get(source_tensor)
        if producer_index is not None and is_slice_operator(model, operators[producer_index]):
            continue

        if not all(is_slice_operator(model, operators[operator_index]) for operator_index in consumer_indices):
            continue

        slice_indices = tuple(consumer_indices)
        moved_indices.update(slice_indices)
        groups.append((slice_indices[0], producer_index, slice_indices))

    if not groups:
        return SubgraphSummary(moved_ops=0, moved_groups=0, operator_count=len(operators))

    groups.sort()
    groups_by_anchor = defaultdict(list)
    for _, producer_index, slice_indices in groups:
        groups_by_anchor[producer_index].append(slice_indices)

    reordered = []

    for slice_indices in groups_by_anchor.get(None, []):
        reordered.extend(operators[index] for index in slice_indices)

    for operator_index, operator in enumerate(operators):
        if operator_index in moved_indices:
            continue

        reordered.append(operator)
        for slice_indices in groups_by_anchor.get(operator_index, []):
            reordered.extend(operators[index] for index in slice_indices)

    if len(reordered) != len(operators):
        raise ValueError("reordered operator count does not match original operator count")

    subgraph["operators"] = reordered
    moved_ops = sum(len(slice_indices) for _, _, slice_indices in groups)
    return SubgraphSummary(
        moved_ops=moved_ops,
        moved_groups=len(groups),
        operator_count=len(operators),
    )


def main():
    args = parse_args()
    tool_dir = Path(__file__).resolve().parent

    schema_path = tool_dir / "schema.fbs"
    assert schema_path.is_file()

    flatc_path = tool_dir / "flatc"
    assert flatc_path.is_file()

    input_path = Path(args.input_tflite).resolve()
    output_path = Path(args.output_tflite).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        json_path = tmpdir / input_path.with_suffix(".json").name

        subprocess.run(
            [flatc_path, "--json", "--strict-json", "-o", str(tmpdir), str(schema_path), "--", str(input_path)],
            check=True,
        )

        with json_path.open("r", encoding="utf-8") as handle:
            model = json.load(handle)

        summaries = [reorder_subgraph(model, subgraph) for subgraph in model.get("subgraphs", [])]

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=2)
            handle.write("\n")

        subprocess.run(
            [flatc_path, "--binary", "-o", str(tmpdir), str(schema_path), str(json_path)],
            check=True,
        )
        shutil.copy2(json_path.with_suffix(".tflite"), output_path)

    if not summaries:
        print("Model has no subgraphs.")
    else:
        total_groups = 0
        for subgraph_index, summary in enumerate(summaries):
            total_groups += summary.moved_groups
            print(
                f"subgraph {subgraph_index}: moved {summary.moved_ops} slice ops "
                f"across {summary.moved_groups} groups ({summary.operator_count} operators)"
            )
        if total_groups == 0:
            print("No reorderable entry slices found; wrote an unchanged model.")

    print(f"Wrote reordered model to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
