from collections import namedtuple

from .conv import _conv_input_slice_for_output_2d, _parse_conv_spec, _parse_pool_spec

_IDENTITY_RANGE_OPS = {"Relu", "Add", "Concat", "Reshape"}

_RangePlan = namedtuple(
    "_RangePlan",
    [
        "split_keys",
        "entry_ranges",
        "output_ranges_by_node",
        "input_ranges_by_node",
        "spatial_slices_by_node",
        "sink_stitch_ranges",
    ],
)


def _partition_ranges(total, part_count):
    base = total // part_count
    rem = total % part_count
    ranges = []
    start = 0
    for part_index in range(part_count):
        size = base + (1 if part_index < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


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


def _plan_node_ranges(group_info):
    split_count_h, split_count_w = group_info.tile_count
    split_keys = [
        (split_id_h, split_id_w)
        for split_id_h in range(split_count_h)
        for split_id_w in range(split_count_w)
    ]
    split_count = len(split_keys)
    node_count = len(group_info.nodes)

    output_ranges_by_node = [[None for _ in range(split_count)] for _ in range(node_count)]
    input_ranges_by_node = [{} for _ in range(node_count)]
    spatial_slices_by_node = [None for _ in range(node_count)]
    entry_ranges = [None for _ in range(split_count)]
    height_ranges = _partition_ranges(group_info.exit_tensor.shape[2], split_count_h)
    width_ranges = _partition_ranges(group_info.exit_tensor.shape[3], split_count_w)
    sink_stitch_ranges = [(h_range, w_range) for h_range in height_ranges for w_range in width_ranges]
    output_ranges_by_node[-1] = _clone_ranges(sink_stitch_ranges)

    for local_index in range(node_count - 1, -1, -1):
        node_spec = group_info.node_specs[local_index]
        node = node_spec.node

        out_ranges = output_ranges_by_node[local_index]
        assert all(rng is not None for rng in out_ranges), (
            f"internal error: missing required ranges for node {node.name} output"
        )

        if node.op == "Conv":
            assert len(node_spec.input_sources) == 1, (
                f"Conv node {node.name} must have exactly one non-constant data input"
            )
            main_input_index = next(iter(node_spec.input_sources))
            spec = _parse_conv_spec(node)
            h_in = node.inputs[main_input_index].shape[2]
            w_in = node.inputs[main_input_index].shape[3]

            demanded_ranges = []
            conv_slices = []
            for (y0, y1), (x0, x1) in out_ranges:
                slice_info = _conv_input_slice_for_output_2d(y0, y1, x0, x1, spec, h_in, w_in)
                demanded_ranges.append(
                    (
                        (slice_info.height.slice_start, slice_info.height.slice_end),
                        (slice_info.width.slice_start, slice_info.width.slice_end),
                    )
                )
                conv_slices.append(slice_info)

            demanded_ranges_by_input = {main_input_index: demanded_ranges}
            spatial_slices_by_node[local_index] = conv_slices
        elif node.op == "AveragePool":
            assert len(node_spec.input_sources) == 1, (
                f"AveragePool node {node.name} must have exactly one non-constant data input"
            )
            main_input_index = next(iter(node_spec.input_sources))
            spec = _parse_pool_spec(node)
            h_in = node.inputs[main_input_index].shape[2]
            w_in = node.inputs[main_input_index].shape[3]

            demanded_ranges = []
            pool_slices = []
            for (y0, y1), (x0, x1) in out_ranges:
                slice_info = _conv_input_slice_for_output_2d(y0, y1, x0, x1, spec, h_in, w_in)
                demanded_ranges.append(
                    (
                        (slice_info.height.slice_start, slice_info.height.slice_end),
                        (slice_info.width.slice_start, slice_info.width.slice_end),
                    )
                )
                pool_slices.append(slice_info)

            demanded_ranges_by_input = {main_input_index: demanded_ranges}
            spatial_slices_by_node[local_index] = pool_slices
        elif node.op in _IDENTITY_RANGE_OPS:
            demanded_ranges_by_input = {
                input_index: _clone_ranges(out_ranges)
                for input_index in node_spec.input_sources
            }
        else:
            assert False, f"unsupported op {node.op} for tiled rewrite planning"

        input_ranges_by_node[local_index] = demanded_ranges_by_input

        for input_index, demanded_ranges in demanded_ranges_by_input.items():
            source = node_spec.input_sources[input_index]
            if source.kind == "entry":
                _merge_range_list(entry_ranges, demanded_ranges)
            else:
                producer_out_ranges = output_ranges_by_node[source.producer_local_index]
                _merge_range_list(producer_out_ranges, demanded_ranges)

    assert all(rng is not None for rng in entry_ranges), (
        f"internal error: missing required ranges for group entry tensor {group_info.entry_tensor.name}"
    )

    for local_index, node_spec in enumerate(group_info.node_specs):
        assert all(rng is not None for rng in output_ranges_by_node[local_index]), (
            f"internal error: missing required ranges for node {node_spec.node.name} output"
        )

    return _RangePlan(
        split_keys=split_keys,
        entry_ranges=entry_ranges,
        output_ranges_by_node=output_ranges_by_node,
        input_ranges_by_node=input_ranges_by_node,
        spatial_slices_by_node=spatial_slices_by_node,
        sink_stitch_ranges=sink_stitch_ranges,
    )
