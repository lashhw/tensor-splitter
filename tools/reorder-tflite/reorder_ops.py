#!/usr/bin/env python3
"""Reorder TFLite operators by round-tripping through FlatBuffers JSON."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Reorder TFLite operators.")
    parser.add_argument("model", help="Input .tflite model.")
    parser.add_argument("--order", required=True, help="File with one operator index per line (blank lines ignored).")
    parser.add_argument("--subgraph", type=int, default=0, help="Subgraph index to reorder.")
    args = parser.parse_args()

    order = []
    with open(args.order, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                order.append(int(line))

    flatc = "./flatc"
    model_path = os.path.abspath(args.model)
    schema_path = "./schema.fbs"
    out_path = os.path.abspath(f"{os.path.splitext(model_path)[0]}_reordered.tflite")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, os.path.basename(model_path).replace(".tflite", ".json"))
        subprocess.run([flatc, "--json", "--strict-json", "-o", tmpdir, schema_path, "--", model_path], check=True)

        with open(json_path, "r", encoding="utf-8") as handle:
            model = json.load(handle)

        ops = model["subgraphs"][args.subgraph]["operators"]
        if len(order) != len(ops):
            raise ValueError("Order length must match operator count.")
        model["subgraphs"][args.subgraph]["operators"] = [ops[i] for i in order]

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=2)
            handle.write("\n")

        subprocess.run([flatc, "--binary", "-o", tmpdir, schema_path, json_path], check=True)
        shutil.copy2(json_path.replace(".json", ".tflite"), out_path)

    print(f"Wrote reordered model to {out_path}")


if __name__ == "__main__":
    main()
