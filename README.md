# ONNX Tensor Splitter

This tool rewrites ONNX models so selected node ranges run as tiles instead of full tensors. It is designed for Conv-heavy linear chains and adds explicit halo handling to preserve numerical results.

## What this tool does

Given an input model and a split plan, the tool:

- Splits the chain input tensor into multiple tiles.
- Rewrites supported operators to run per tile.
- Inserts/adjusts halo overlap for convolution correctness.
- Reassembles the tiled results back into the original tensor layout.

## Requirements

```bash
python -m venv venv
source venv/bin/activate
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

## Configuration format

The config must be a **JSON list** of group entries.

Each group contains:

- `node_range`: `[start_node_index, end_node_index]` (inclusive)
- `tile_count`: number of tiles
- `execution_order`: list of `[node_index, split_id]` pairs

### Example

```json
[
  {
    "node_range": [4, 6],
    "tile_count": 2,
    "execution_order": [[4, 0], [4, 1], [5, 0], [5, 1], [6, 0], [6, 1]]
  },
  {
    "node_range": [10, 11],
    "tile_count": 3,
    "execution_order": [[10, 0], [11, 0], [10, 1], [11, 1], [10, 2], [11, 2]]
  }
]
```

### Validation rules

- `node_range` must be non-negative and `start <= end`.
- `tile_count` must be `> 0`.
- `execution_order` must include every `(node_index, split_id)` in the group exactly once.
- Group ranges must be disjoint.

## Testing

Run the test suite:

```bash
pytest -q
```

## Project layout

- `split_onnx.py`: CLI entry point.
- `src/config.py`: config parsing and validation.
- `src/tiling.py`: tensor-range and convolution halo math.
- `src/utils.py`: group analysis and graph constraints.
- `src/rewrite.py`: compatibility shim exporting `rewrite_model`.
- `src/rewriter/rewriter.py`: high-level model rewrite pipeline.
- `src/rewriter/tile_builders.py`: op-specific tiled graph construction.
- `src/rewriter/tensor_ops.py`: graph tensor helpers (slice/concat/pad/conv attrs).
- `src/rewriter/graph_utils.py`: graph ordering and consumer rewiring utilities.
- `src/rewriter/types.py`: small shared data structures/constants.
