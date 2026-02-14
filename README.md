# ONNX Tensor Splitter

This tool rewrites ONNX models so selected node ranges run as tiles instead of full tensors. It is designed for Conv-heavy linear chains and adds explicit halo handling to preserve numerical results.

## What this tool does

Given an ONNX model and a split config, this tool generates a new ONNX model where selected node ranges run tile-by-tile instead of on full tensors.

For each configured node range, it:

1. Splits the range input tensor into `tile_count` tiles.
2. Computes the halo each tile needs so convolution results match the original full-tensor run.
3. Rewrites supported operators in that range to consume and produce per-tile tensors.
4. Applies convolution border padding per tile when needed.
5. Stitches tile outputs back into the original tensor layout.

The rewritten model keeps the same external I/O interface, and the CLI verifies numerical equivalence with ONNX Runtime by default.

## Quick start

1. Create and activate a virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Create a split config JSON file (see [Configuration format](#configuration-format)) and run the rewrite command.

   ```bash
   python -m ts.cli input.onnx config.json output.onnx
   ```

4. Check the verification result (`PASS` or `FAIL`) printed by the CLI.

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
- `execution_order`: list of `[node_index, split_id]` pairs (must include each pair in the group exactly once)

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

## Testing

Run the test suite:

```bash
pytest
```
