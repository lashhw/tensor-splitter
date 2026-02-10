# ONNX Tensor Splitter (Height Tiling)

`tensor-splitter` rewrites ONNX models so selected node ranges run as **height tiles** (NCHW axis `2`) instead of full-height tensors.  
It is designed for Conv-heavy linear chains and adds explicit halo handling to preserve numerical results.

## What this tool does

Given an input model and a split plan, the tool:

- Splits the chain input tensor into `N` height tiles.
- Rewrites supported operators to run per tile.
- Inserts/adjusts halo overlap for convolution correctness.
- Reassembles the tiled results back into the original tensor layout.
- Stores the execution schedule in model metadata (`tensor_split_schedule`).

## Requirements

- Python 3.9+
- `onnx`
- `onnxruntime` (only required for `--verify`)
- `onnx-graphsurgeon`

Install dependencies:

```bash
pip install -r requirements.txt
# plus graphsurgeon if not already available in your environment
pip install onnx-graphsurgeon
```

## Quick start

1. Prepare a JSON config file (see [Configuration format](#configuration-format)).
2. Run the rewrite:

```bash
python3 split_onnx.py \
  --model model.onnx \
  --config split_config.json \
  --output split_model.onnx
```

3. (Optional) Run numeric verification against the original model:

```bash
python3 split_onnx.py \
  --model model.onnx \
  --config split_config.json \
  --output split_model.onnx \
  --verify --verbose
```

## CLI reference

```text
--model   Path to input ONNX model (required)
--config  Path to split configuration JSON (required)
--output  Path to output ONNX model (required)
--verify  Run onnxruntime-based output comparison
--verbose Enable verbose rewrite logging
```

## Configuration format

The config must be a **JSON list** of group entries.

Each group contains:

- `indices`: `[start_node_index, end_node_index]` (inclusive)
- `splits`: number of height tiles
- `schedule`: list of `[node_index, split_id]` pairs

### Example

```json
[
  {
    "indices": [4, 6],
    "splits": 2,
    "schedule": [[4, 0], [4, 1], [5, 0], [5, 1], [6, 0], [6, 1]]
  },
  {
    "indices": [10, 11],
    "splits": 3,
    "schedule": [[10, 0], [11, 0], [10, 1], [11, 1], [10, 2], [11, 2]]
  }
]
```

### Validation rules

- `indices` must be non-negative and `start <= end`.
- `splits` must be `> 0`.
- `schedule` must include every `(node_index, split_id)` in the group exactly once.
- Group ranges must be disjoint and non-touching.

## Supported operators (v1)

- `Conv`
- Unary: `Relu`, `Sigmoid`, `Tanh`, `Identity`
- Unary with constant inputs: `Clip`, `BatchNormalization`
- Binary: `Add`, `Mul`, `Sub`, `Div` (external non-constant inputs are not supported)

## Current constraints

- Split axis is fixed to Height (NCHW axis `2`).
- Each group must be a **linear chain**:
  - single data input into first node,
  - each node consumes the previous node output as its only data input,
  - single exit tensor (output of node `end`).
- Height must be statically known after shape inference.
- `Conv` with `auto_pad` is not supported.

## Testing

Run the test suite:

```bash
pytest -q
```

## Troubleshooting

- **Config parse error**: ensure config is valid JSON (not Python literal syntax).
- **Schedule mismatch error**: verify every pair `(node_index, split_id)` appears exactly once.
- **Verification failure**: rerun with `--verbose`, then inspect unsupported patterns in selected node ranges.
