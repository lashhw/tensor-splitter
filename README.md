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

Assuming dependencies are installed (see [Requirements](#requirements)):

1. Create a split config JSON file (see [Configuration format](#configuration-format)).
2. Run the rewrite command.
3. Check the verification result (`PASS` or `FAIL`) printed by the CLI.

```bash
python -m tensor_splitter.cli model.onnx split_config.json model_tiled.onnx
```

## CLI reference

### Usage

```bash
python -m tensor_splitter.cli INPUT CONFIG OUTPUT [--verify | --no-verify]
```

### Positional arguments

| Name | Description |
| --- | --- |
| `INPUT` | Path to input ONNX model. |
| `CONFIG` | Path to split configuration JSON. |
| `OUTPUT` | Path where the rewritten ONNX model will be saved. |

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--verify` / `--no-verify` | Enabled | Enable or disable ONNX Runtime numerical comparison after rewriting. |

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

- `tensor_splitter/cli.py`: CLI entry point.
- `tensor_splitter/config.py`: config parsing and validation.
- `tensor_splitter/verification.py`: numerical output comparison with ONNX Runtime.
- `tensor_splitter/analysis/group_chain.py`: group analysis and graph constraints.
- `tensor_splitter/tiling/geometry.py`: tensor-range and convolution halo math.
- `tensor_splitter/rewrite/pipeline.py`: high-level model rewrite pipeline.
- `tensor_splitter/rewrite/op_rewriters.py`: op-specific tiled graph construction.
- `tensor_splitter/rewrite/graph_ops.py`: graph tensor helpers (slice/concat/pad/conv attrs).
- `tensor_splitter/rewrite/scheduling.py`: graph ordering and consumer rewiring utilities.
- `tensor_splitter/rewrite/models.py`: tile block model and supported op sets.
- `tensor_splitter/rewrite/naming.py`: graph-safe rewritten name generation.
