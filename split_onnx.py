#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

import onnx

from src.config import ConfigError, parse_config
from src.rewrite import RewriteError, rewrite_model
from src.verify import verify_models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split ONNX model spatially along Height axis")
    parser.add_argument("--model", required=True, help="Path to input ONNX model")
    parser.add_argument("--config", required=True, help="Path to config (py/json/txt)")
    parser.add_argument("--output", required=True, help="Path to output ONNX model")
    parser.add_argument("--verify", action="store_true", help="Run onnxruntime verification")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        groups = parse_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    model = onnx.load(args.model)

    try:
        rewritten, stats = rewrite_model(model, groups, verbose=args.verbose)
    except RewriteError as exc:
        print(f"Rewrite error: {exc}", file=sys.stderr)
        return 3

    onnx.save(rewritten, args.output)

    print(
        json.dumps(
            {
                "groups_rewritten": stats.groups_rewritten,
                "main_nodes_cloned": stats.main_nodes_cloned,
                "new_nodes_inserted": stats.new_nodes_inserted,
            },
            indent=2,
        )
    )

    if args.verify:
        ok, diffs = verify_models(model, rewritten)
        print("Verification: PASS" if ok else "Verification: FAIL")
        for name, diff in diffs.items():
            print(f"{name}: max_abs_diff={diff}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
