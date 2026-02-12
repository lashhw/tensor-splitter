from __future__ import annotations

import argparse

import onnx

from ts.config import parse_config
from ts.rewrite import rewrite_model
from ts.verify import verify_models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input ONNX model")
    parser.add_argument("config", help="Path to split configuration JSON")
    parser.add_argument("output", help="Path to output ONNX model")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNX Runtime numerical verification (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    groups = parse_config(args.config)
    model = onnx.load(args.input)
    rewritten = rewrite_model(model, groups)
    onnx.save(rewritten, args.output)

    if args.verify:
        ok, diffs = verify_models(model, rewritten)
        print("Verification: PASS" if ok else "Verification: FAIL")
        for name, diff in diffs.items():
            print(f"{name}: max_abs_diff={diff}")


if __name__ == "__main__":
    main()
