import argparse

import onnx

from tensor_splitter.config import parse_config
from tensor_splitter.rewrite import rewrite_model
from tensor_splitter.verification import verify_models


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to input ONNX model")
    parser.add_argument("--config", required=True, help="Path to config (json)")
    parser.add_argument("--output", required=True, help="Path to output ONNX model")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run onnxruntime verification (default: enabled)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    groups = parse_config(args.config)

    model = onnx.load(args.model)

    rewritten = rewrite_model(model, groups)

    onnx.save(rewritten, args.output)

    if args.verify:
        ok, diffs = verify_models(model, rewritten)
        print("Verification: PASS" if ok else "Verification: FAIL")
        for name, diff in diffs.items():
            print(f"{name}: max_abs_diff={diff}")


if __name__ == "__main__":
    main()
