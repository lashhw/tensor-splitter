# ONNX Tensor Splitter

`tensor-splitter` rewrites ONNX models so selected node ranges run as tiles instead of full tensors. It is designed for Conv-heavy linear chains and adds explicit halo handling to preserve numerical results.

## What this tool does

Given an input model and a split plan, the tool:

- Splits the chain input tensor into `N` tiles.
- Rewrites supported operators to run per tile.
- Inserts/adjusts halo overlap for convolution correctness.
- Reassembles the tiled results back into the original tensor layout.

## Requirements

```bash
pip install -r requirements.txt
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
  --verify
```

## CLI reference

```text
--model   Path to input ONNX model (required)
--config  Path to split configuration JSON (required)
--output  Path to output ONNX model (required)
--verify  Run onnxruntime-based output comparison
```

## Project layout

- `split_onnx.py` CLI entry point (thin wrapper around `src/cli.py`)
- `src/cli.py` argument parsing + model IO
- `src/rewrite/` graph rewrite pipeline (tiling, node synthesis, graph assembly)
- `src/group_analysis.py` linear-chain group validation and metadata extraction
- `src/tiling.py` convolution tiling math helpers
- `src/config.py` config parsing + validation
- `src/verify.py` onnxruntime-based verification
- `tests/` unit tests

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

## Testing

Run the test suite:

```bash
pytest -q
```
