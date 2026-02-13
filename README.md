# ONNX Tensor Splitter

This tool rewrites ONNX models so selected node ranges run as tiles instead of full tensors. It is designed for Conv-heavy linear chains and adds explicit halo handling to preserve numerical results.

## What this tool does

Given an input model and a split plan, the tool:

- Splits the chain input tensor into multiple tiles.
- Back-propagates required ranges from group output to build per-tile entry halos.
- Rewrites supported operators to run per tile inside the group.
- Handles convolution border padding locally per tile.
- Reassembles the tiled results back into the original tensor layout.

## Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick start

1. Create a split config JSON file (see [Configuration format](#configuration-format)).
2. Run the rewrite command.

   ```bash
   python -m ts.cli input.onnx config.json output.onnx
   ```

3. Check the verification result (`PASS` or `FAIL`) printed by the CLI.

## CLI reference

### Usage

```bash
python -m ts.cli INPUT CONFIG OUTPUT [--no-verify]
```

### Arguments

| Name | Description |
| --- | --- |
| `INPUT` | Path to input ONNX model. |
| `CONFIG` | Path to split configuration JSON. |
| `OUTPUT` | Path to output ONNX model. |
| `--no-verify` | Disable ONNX Runtime numerical verification. |

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
pytest
```
