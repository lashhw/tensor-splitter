# ONNX Tensor Splitter (Height Tiling)

This tool rewrites an ONNX model to perform spatial tensor splitting along the Height axis (NCHW axis=2) for specified node index ranges. It focuses on Conv-heavy linear chains and preserves numerical results by introducing explicit halo handling.

## Usage

```bash
python3 split_onnx.py --model model.onnx --config split_config.py --output split_model.onnx
```

Optional flags:

```bash
python3 split_onnx.py --model model.onnx --config split_config.json --output split_model.onnx --verify --verbose
```

## Config format

Config is a list of dict-like entries. Example (Python literal):

```python
[
  {
    "indices": (4, 6),
    "splits": 2,
    "schedule": [(4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1)]
  },
  {
    "indices": (10, 11),
    "splits": 3,
    "schedule": [(10, 0), (11, 0), (10, 1), (11, 1), (10, 2), (11, 2)]
  }
]
```

JSON is also accepted (use lists instead of tuples). The tool stores the schedule as model metadata under `tensor_split_schedule`.

## Supported ops (v1)

- `Conv`
- Unary: `Relu`, `Sigmoid`, `Tanh`, `Identity`
- Unary with constant inputs: `Clip`, `BatchNormalization`
- Binary: `Add`, `Mul`, `Sub`, `Div` (external inputs must be constants)

## Constraints (v1)

- Split axis is Height (NCHW axis=2).
- Each group must be a linear chain: a single data input into the first node, and each node consumes the previous node output as its only data input.
- The group must have a single exit tensor (output of node b).
- External variable inputs (besides the chain input) are not supported.
- Height must be statically known after shape inference.
- `Conv` with `auto_pad` is not supported.

## Tests

```bash
pytest -q
```

## Notes

- The tool uses the local TensorRT checkout’s `onnx-graphsurgeon` package by default.
- Verification uses onnxruntime CPU.
