import numpy as np
import onnx_graphsurgeon as gs


def _is_constant(tensor):
    return isinstance(tensor, gs.Constant)


def _tensor_height(tensor):
    return tensor.shape[2]


def _shape_with_dim_size(
    shape,
    dim,
    size,
):
    new_shape = list(shape)
    new_shape[dim] = size
    return new_shape


def _make_constant(name_scope, values):
    return gs.Constant(name_scope.make("tsplit_const"), values)


def _make_slice(
    name_scope,
    data,
    start,
    end,
    axis,
):
    starts = _make_constant(name_scope, np.array([start], dtype=np.int64))
    ends = _make_constant(name_scope, np.array([end], dtype=np.int64))
    axes = _make_constant(name_scope, np.array([axis], dtype=np.int64))
    steps = _make_constant(name_scope, np.array([1], dtype=np.int64))
    out = gs.Variable(
        name_scope.make(f"{data.name}_slice"),
        dtype=data.dtype,
        shape=_shape_with_dim_size(data.shape, axis, end - start),
    )
    node = gs.Node(op="Slice", inputs=[data, starts, ends, axes, steps], outputs=[out])
    return out, node


def _make_concat(
    name_scope,
    inputs,
    axis,
    shape,
):
    out_shape = list(shape)
    out = gs.Variable(name_scope.make("tsplit_concat"), dtype=inputs[0].dtype, shape=out_shape)
    node = gs.Node(op="Concat", inputs=inputs, outputs=[out], attrs={"axis": axis})
    return out, node
