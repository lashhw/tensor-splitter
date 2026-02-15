import onnx_graphsurgeon as gs


def _is_constant(tensor):
    return isinstance(tensor, gs.Constant)


def _tensor_height(tensor):
    return tensor.shape[2]


def _shape_with_dim_size(shape, dim, size):
    new_shape = list(shape)
    new_shape[dim] = size
    return new_shape
