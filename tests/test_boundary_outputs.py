from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from ts.rewrite import rewrite_model
from ts.verify import verify_model


def _make_group(node_range, tile_count, execution_order):
    return SimpleNamespace(
        node_range=node_range,
        tile_count=tile_count,
        execution_order=execution_order,
    )


def _make_boundary_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 7, 7])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 7, 7])

    w1 = numpy_helper.from_array(
        (np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3) / 100.0),
        name="w1",
    )
    w2 = numpy_helper.from_array(
        (np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3) / 200.0),
        name="w2",
    )

    conv_attrs = {
        "kernel_shape": [3, 3],
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
        "dilations": [1, 1],
    }
    nodes = [
        helper.make_node("Relu", inputs=["x"], outputs=["n10_out"], name="n10"),
        helper.make_node("Conv", inputs=["n10_out", "w1"], outputs=["n11_out"], name="n11", **conv_attrs),
        helper.make_node("Relu", inputs=["n11_out"], outputs=["n12_out"], name="n12"),
        helper.make_node("Conv", inputs=["n12_out", "w2"], outputs=["n13_out"], name="n13", **conv_attrs),
        helper.make_node("Relu", inputs=["n12_out"], outputs=["n14_out"], name="n14"),
        helper.make_node("Add", inputs=["n13_out", "n14_out"], outputs=["y"], name="n15"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="boundary_model",
        inputs=[x],
        outputs=[y],
        initializer=[w1, w2],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    onnx.checker.check_model(model)
    return model


def _make_two_group_chain_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 6, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 6, 6])

    nodes = [
        helper.make_node("Relu", inputs=["x"], outputs=["n10_out"], name="n10"),
        helper.make_node("Relu", inputs=["n10_out"], outputs=["n11_out"], name="n11"),
        helper.make_node("Relu", inputs=["n11_out"], outputs=["n12_out"], name="n12"),
        helper.make_node("Relu", inputs=["n12_out"], outputs=["n13_out"], name="n13"),
        helper.make_node("Relu", inputs=["n13_out"], outputs=["n14_out"], name="n14"),
        helper.make_node("Relu", inputs=["n14_out"], outputs=["y"], name="n15"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="two_group_chain_model",
        inputs=[x],
        outputs=[y],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    onnx.checker.check_model(model)
    return model


def _boundary_group():
    return _make_group(
        node_range=(1, 3),
        tile_count=(2, 1),
        execution_order=[
            (1, (0, 0)),
            (2, (0, 0)),
            (3, (0, 0)),
            (1, (1, 0)),
            (2, (1, 0)),
            (3, (1, 0)),
        ],
    )


def test_rewrite_supports_boundary_output_to_nonsplit_node():
    model = _make_boundary_model()
    rewritten = rewrite_model(model, [_boundary_group()])

    ok, diffs = verify_model(model, rewritten)
    assert ok, f"rewritten model output mismatch: {diffs}"


def test_rewrite_adds_boundary_stitch_crop_and_concat_nodes():
    model = _make_boundary_model()
    rewritten = rewrite_model(model, [_boundary_group()])

    rewritten_nodes = list(rewritten.graph.node)
    assert any(node.name == "n12_out_concat" and node.op_type == "Concat" for node in rewritten_nodes)
    assert any(
        node.op_type == "Slice" and "n12_out_stitch_l1_crop" in node.name
        for node in rewritten_nodes
    )


def test_rewrite_supports_adjacent_split_ranges_with_cross_range_edge():
    model = _make_two_group_chain_model()
    groups = [
        _make_group(
            node_range=(1, 2),
            tile_count=(2, 1),
            execution_order=[
                (1, (0, 0)),
                (2, (0, 0)),
                (1, (1, 0)),
                (2, (1, 0)),
            ],
        ),
        _make_group(
            node_range=(3, 4),
            tile_count=(2, 1),
            execution_order=[
                (3, (0, 0)),
                (4, (0, 0)),
                (3, (1, 0)),
                (4, (1, 0)),
            ],
        ),
    ]

    rewritten = rewrite_model(model, groups)
    ok, diffs = verify_model(model, rewritten)
    assert ok, f"rewritten model output mismatch: {diffs}"
