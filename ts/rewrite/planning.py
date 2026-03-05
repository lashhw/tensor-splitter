from collections import namedtuple

from .conv import _conv_input_slice_for_output_2d, _parse_conv_spec, _parse_pool_spec
from .tensor import _tensor_height, _tensor_width

_RangePlan = namedtuple(
    "_RangePlan",
    [
        "split_keys",
        "entry_ranges_by_key",
        "output_ranges_by_node",
        "input_ranges_by_node",
        "spatial_slices_by_node",
    ],
)


def _partition_ranges(total, tile_count):
    base = total // tile_count
    rem = total % tile_count
    ranges = []
    start = 0
    for i in range(tile_count):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def _split_keys(tile_count):
    split_count_h, split_count_w = tile_count
    return [(split_id_h, split_id_w) for split_id_h in range(split_count_h) for split_id_w in range(split_count_w)]


def _partition_ranges_2d(height, width, tile_count):
    split_count_h, split_count_w = tile_count
    height_ranges = _partition_ranges(height, split_count_h)
    width_ranges = _partition_ranges(width, split_count_w)
    return [(h_range, w_range) for h_range in height_ranges for w_range in width_ranges]


def _merge_range(existing, new_range):
    if existing is None:
        return new_range
    (y0, y1), (x0, x1) = existing
    (ny0, ny1), (nx0, nx1) = new_range
    return ((min(y0, ny0), max(y1, ny1)), (min(x0, nx0), max(x1, nx1)))


def _merge_range_list(dst_ranges, src_ranges):
    assert len(dst_ranges) == len(src_ranges), "internal error: range list size mismatch"
    for idx, src_range in enumerate(src_ranges):
        assert src_range is not None, "internal error: source range must be initialized"
        dst_ranges[idx] = _merge_range(dst_ranges[idx], src_range)


def _clone_ranges(ranges):
    return [((y0, y1), (x0, x1)) for (y0, y1), (x0, x1) in ranges]


def _require_fully_initialized(ranges, context):
    assert all(rng is not None for rng in ranges), f"internal error: missing required ranges for {context}"


def _propagate_identity_input_ranges(out_ranges, data_input_indices):
    return {input_index: _clone_ranges(out_ranges) for input_index in data_input_indices}


def _propagate_conv_input_ranges(node, node_spec, out_ranges):
    assert len(node_spec.data_input_indices) == 1, (
        f"Conv node {node.name} must have exactly one non-constant data input"
    )
    main_input_index = node_spec.data_input_indices[0]
    spec = _parse_conv_spec(node)
    h_in = _tensor_height(node.inputs[main_input_index])
    w_in = _tensor_width(node.inputs[main_input_index])

    in_ranges = []
    conv_slices = []
    for (y0, y1), (x0, x1) in out_ranges:
        slice_info = _conv_input_slice_for_output_2d(y0, y1, x0, x1, spec, h_in, w_in)
        in_ranges.append(
            (
                (slice_info.height.slice_start, slice_info.height.slice_end),
                (slice_info.width.slice_start, slice_info.width.slice_end),
            )
        )
        conv_slices.append(slice_info)

    return {main_input_index: in_ranges}, conv_slices


def _propagate_pool_input_ranges(node, node_spec, out_ranges):
    assert len(node_spec.data_input_indices) == 1, (
        f"AveragePool node {node.name} must have exactly one non-constant data input"
    )
    main_input_index = node_spec.data_input_indices[0]
    spec = _parse_pool_spec(node)
    h_in = _tensor_height(node.inputs[main_input_index])
    w_in = _tensor_width(node.inputs[main_input_index])

    in_ranges = []
    pool_slices = []
    for (y0, y1), (x0, x1) in out_ranges:
        slice_info = _conv_input_slice_for_output_2d(y0, y1, x0, x1, spec, h_in, w_in)
        in_ranges.append(
            (
                (slice_info.height.slice_start, slice_info.height.slice_end),
                (slice_info.width.slice_start, slice_info.width.slice_end),
            )
        )
        pool_slices.append(slice_info)

    return {main_input_index: in_ranges}, pool_slices


def _plan_node_ranges(group_info, tile_count):
    split_keys = _split_keys(tile_count)
    split_count = len(split_keys)
    node_count = len(group_info.nodes)

    output_ranges_by_node = [[None for _ in range(split_count)] for _ in range(node_count)]
    input_ranges_by_node = [{} for _ in range(node_count)]
    spatial_slices_by_node = [None for _ in range(node_count)]
    entry_ranges_by_key = {
        entry_key: [None for _ in range(split_count)]
        for entry_key in group_info.entry_tensors
    }

    output_ranges_by_node[-1] = _partition_ranges_2d(
        height=_tensor_height(group_info.exit_tensor),
        width=_tensor_width(group_info.exit_tensor),
        tile_count=tile_count,
    )

    for local_index in range(node_count - 1, -1, -1):
        node_spec = group_info.node_specs[local_index]
        node = node_spec.node

        if node.op == "Constant":
            input_ranges_by_node[local_index] = {}
            continue

        out_ranges = output_ranges_by_node[local_index]
        _require_fully_initialized(out_ranges, f"node {node.name} output")

        if node.op == "Conv":
            demanded_ranges_by_input, conv_slices = _propagate_conv_input_ranges(node, node_spec, out_ranges)
            spatial_slices_by_node[local_index] = conv_slices
        elif node.op == "AveragePool":
            demanded_ranges_by_input, pool_slices = _propagate_pool_input_ranges(node, node_spec, out_ranges)
            spatial_slices_by_node[local_index] = pool_slices
        elif node.op in {"Relu", "BatchNormalization", "Add", "Concat", "Reshape", "Transpose"}:
            demanded_ranges_by_input = _propagate_identity_input_ranges(
                out_ranges,
                node_spec.data_input_indices,
            )
        else:
            assert False, f"unsupported op {node.op} for tiled rewrite planning"

        input_ranges_by_node[local_index] = demanded_ranges_by_input

        for input_index, demanded_ranges in demanded_ranges_by_input.items():
            source = node_spec.input_sources[input_index]
            if source.kind == "entry":
                entry_ranges = entry_ranges_by_key[source.entry_key]
                _merge_range_list(entry_ranges, demanded_ranges)
            else:
                producer_out_ranges = output_ranges_by_node[source.producer_local_index]
                _merge_range_list(producer_out_ranges, demanded_ranges)

    for entry_key, entry_ranges in entry_ranges_by_key.items():
        entry_tensor = group_info.entry_tensors[entry_key]
        _require_fully_initialized(entry_ranges, f"group entry tensor {entry_tensor.name}")

    for local_index, node_spec in enumerate(group_info.node_specs):
        if node_spec.node.op == "Constant":
            continue
        _require_fully_initialized(output_ranges_by_node[local_index], f"node {node_spec.node.name} output")

    return _RangePlan(
        split_keys=split_keys,
        entry_ranges_by_key=entry_ranges_by_key,
        output_ranges_by_node=output_ranges_by_node,
        input_ranges_by_node=input_ranges_by_node,
        spatial_slices_by_node=spatial_slices_by_node,
    )
