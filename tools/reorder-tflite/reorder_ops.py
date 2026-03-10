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

SLICE_OP_NAMES = {"SLICE", "STRIDED_SLICE"}
BUILTIN_OPERATOR_NAMES = {
    0: "ADD",
    32: "CUSTOM",
    45: "STRIDED_SLICE",
    65: "SLICE",
}


@dataclass(frozen=True)
class SliceGroup:
    anchor_index: int | None
    source_tensor: int
    operator_indices: tuple[int, ...]
    first_index: int


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


def resolve_flatc(tool_dir: Path) -> str:
    local_flatc = tool_dir / "flatc"
    assert local_flatc.is_file()
    return str(local_flatc)


def operator_code_name(model, opcode_index: int) -> str:
    code = model["operator_codes"][opcode_index]
    builtin_name = code.get("builtin_code")
    if builtin_name:
        return builtin_name
    deprecated_code = code.get("deprecated_builtin_code", 0)
    return BUILTIN_OPERATOR_NAMES.get(deprecated_code, f"BUILTIN_{deprecated_code}")


def operator_name(model, operator) -> str:
    return operator_code_name(model, operator.get("opcode_index", 0))


def is_slice_operator(model, operator) -> bool:
    return operator_name(model, operator) in SLICE_OP_NAMES


def build_tensor_maps(operators):
    producer_by_tensor = {}
    consumers_by_tensor = defaultdict(list)

    for operator_index, operator in enumerate(operators):
        for tensor_index in operator.get("outputs", []):
            if tensor_index >= 0:
                producer_by_tensor[tensor_index] = operator_index

        for tensor_index in operator.get("inputs", []):
            if tensor_index >= 0:
                consumers_by_tensor[tensor_index].append(operator_index)

    return producer_by_tensor, consumers_by_tensor


def is_reorderable_entry_slice(model, operators, operator_index, source_tensor, producer_by_tensor, subgraph_inputs):
    operator = operators[operator_index]
    if not is_slice_operator(model, operator):
        return False

    inputs = operator.get("inputs", [])
    if not inputs or inputs[0] != source_tensor:
        return False

    producer_index = producer_by_tensor.get(source_tensor)
    if producer_index is None:
        return source_tensor in subgraph_inputs

    return not is_slice_operator(model, operators[producer_index])


def find_reorder_groups(model, subgraph):
    operators = subgraph.get("operators", [])
    producer_by_tensor, consumers_by_tensor = build_tensor_maps(operators)
    subgraph_inputs = {tensor_index for tensor_index in subgraph.get("inputs", []) if tensor_index >= 0}
    groups = []

    for source_tensor, consumer_indices in consumers_by_tensor.items():
        slice_indices = [
            operator_index
            for operator_index in consumer_indices
            if is_reorderable_entry_slice(
                model=model,
                operators=operators,
                operator_index=operator_index,
                source_tensor=source_tensor,
                producer_by_tensor=producer_by_tensor,
                subgraph_inputs=subgraph_inputs,
            )
        ]

        if len(slice_indices) < 2:
            continue

        groups.append(
            SliceGroup(
                anchor_index=producer_by_tensor.get(source_tensor),
                source_tensor=source_tensor,
                operator_indices=tuple(slice_indices),
                first_index=slice_indices[0],
            )
        )

    groups.sort(key=lambda group: group.first_index)
    return groups


def rebuild_operator_order(operators, groups):
    groups_by_anchor = defaultdict(list)
    moved_indices = set()

    for group in groups:
        overlap = moved_indices.intersection(group.operator_indices)
        if overlap:
            overlap_text = ", ".join(str(index) for index in sorted(overlap))
            raise ValueError(f"slice group overlap detected for operators: {overlap_text}")

        moved_indices.update(group.operator_indices)
        groups_by_anchor[group.anchor_index].append(group)

    reordered = []

    for group in groups_by_anchor.get(None, []):
        reordered.extend(operators[index] for index in group.operator_indices)

    for operator_index, operator in enumerate(operators):
        if operator_index in moved_indices:
            continue

        reordered.append(operator)
        for group in groups_by_anchor.get(operator_index, []):
            reordered.extend(operators[index] for index in group.operator_indices)

    if len(reordered) != len(operators):
        raise ValueError("reordered operator count does not match original operator count")

    return reordered


def reorder_subgraph(model, subgraph):
    operators = subgraph.get("operators", [])
    groups = find_reorder_groups(model, subgraph)
    if groups:
        subgraph["operators"] = rebuild_operator_order(operators, groups)

    moved_ops = sum(len(group.operator_indices) for group in groups)
    return SubgraphSummary(
        moved_ops=moved_ops,
        moved_groups=len(groups),
        operator_count=len(operators),
    )


def reorder_model(model):
    return [reorder_subgraph(model, subgraph) for subgraph in model.get("subgraphs", [])]


def model_json_path(tmpdir: Path, model_path: Path) -> Path:
    return tmpdir / model_path.with_suffix(".json").name


def export_model_json(flatc_path: str, schema_path: Path, model_path: Path, tmpdir: Path) -> Path:
    json_path = model_json_path(tmpdir, model_path)
    subprocess.run(
        [flatc_path, "--json", "--strict-json", "-o", str(tmpdir), str(schema_path), "--", str(model_path)],
        check=True,
    )
    return json_path


def write_model_json(json_path: Path, model):
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2)
        handle.write("\n")


def build_tflite(flatc_path: str, schema_path: Path, json_path: Path, tmpdir: Path) -> Path:
    subprocess.run(
        [flatc_path, "--binary", "-o", str(tmpdir), str(schema_path), str(json_path)],
        check=True,
    )
    return json_path.with_suffix(".tflite")


def print_summary(summaries):
    if not summaries:
        print("Model has no subgraphs.")
        return

    total_groups = 0
    for subgraph_index, summary in enumerate(summaries):
        total_groups += summary.moved_groups
        print(
            f"subgraph {subgraph_index}: moved {summary.moved_ops} slice ops "
            f"across {summary.moved_groups} groups ({summary.operator_count} operators)"
        )

    if total_groups == 0:
        print("No reorderable entry slices found; wrote an unchanged model.")


def main():
    args = parse_args()
    tool_dir = Path(__file__).resolve().parent
    schema_path = tool_dir / "schema.fbs"
    if not schema_path.is_file():
        raise FileNotFoundError(f"TFLite schema not found at {schema_path}")

    flatc_path = resolve_flatc(tool_dir)
    input_path = Path(args.input_tflite).resolve()
    output_path = Path(args.output_tflite).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        json_path = export_model_json(flatc_path, schema_path, input_path, tmpdir)

        with json_path.open("r", encoding="utf-8") as handle:
            model = json.load(handle)

        summaries = reorder_model(model)
        write_model_json(json_path, model)

        rewritten_path = build_tflite(flatc_path, schema_path, json_path, tmpdir)
        shutil.copy2(rewritten_path, output_path)

    print_summary(summaries)
    print(f"Wrote reordered model to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
