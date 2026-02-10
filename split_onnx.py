from __future__ import annotations

import argparse
import onnx

from src.config import parse_config
from src.rewrite import rewrite_model
from src.verify import verify_models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to input ONNX model")
    parser.add_argument("--config", required=True, help="Path to config (json)")
    parser.add_argument("--output", required=True, help="Path to output ONNX model")
    parser.add_argument("--verify", action="store_true", help="Run onnxruntime verification")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    groups = parse_config(args.config)

    model = onnx.load(args.model)
    rewritten = rewrite_model(model, groups, verbose=args.verbose)
    onnx.save(rewritten, args.output)

    if args.verify:
        ok, diffs = verify_models(model, rewritten)
        print("Verification: PASS" if ok else "Verification: FAIL")
        for name, diff in diffs.items():
            print(f"{name}: max_abs_diff={diff}")


if __name__ == "__main__":
    main()
