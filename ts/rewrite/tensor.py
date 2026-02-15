import numpy as np
import onnx_graphsurgeon as gs


def _is_constant(tensor):
    return isinstance(tensor, gs.Constant)


def _tensor_height(tensor):
    return tensor.shape[2]


def _shape_with_dim_size(shape, dim, size):
    new_shape = list(shape)
    new_shape[dim] = size
    return new_shape


def _make_constant(name, values):
    return gs.Constant(name, values)


def _make_slice(data, start, end, axis, tile_id):
    base_name = f"{data.name}_slice{tile_id}"
    starts = _make_constant(f"{base_name}_starts", np.array([start], dtype=np.int64))
    ends = _make_constant(f"{base_name}_ends", np.array([end], dtype=np.int64))
    axes = _make_constant(f"{base_name}_axes", np.array([axis], dtype=np.int64))
    steps = _make_constant(f"{base_name}_steps", np.array([1], dtype=np.int64))
    out = gs.Variable(
        f"{data.name}_split{tile_id}",
        dtype=data.dtype,
        shape=_shape_with_dim_size(data.shape, axis, end - start),
    )
    node = gs.Node(
        name=base_name,
        op="Slice",
        inputs=[data, starts, ends, axes, steps],
        outputs=[out],
    )
    return out, node


def _make_concat(inputs, axis, shape, output_tensor_name):
    out_shape = list(shape)
    out = gs.Variable(output_tensor_name, dtype=inputs[0].dtype, shape=out_shape)
    node = gs.Node(
        name=f"{output_tensor_name}_concat",
        op="Concat",
        inputs=inputs,
        outputs=[out],
        attrs={"axis": axis},
    )
    return out, node
