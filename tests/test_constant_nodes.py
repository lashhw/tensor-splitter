from types import SimpleNamespace
from unittest.mock import patch

import onnx
import onnx_graphsurgeon as gs
import pytest
from onnx import TensorProto, helper

from ts.rewrite import rewrite_model
from ts.rewrite.pipeline import _remove_constant_nodes
from ts.verify import verify_model


def _make_group(node_range, tile_count, execution_order):
    return SimpleNamespace(
        node_range=node_range,
        tile_count=tile_count,
        execution_order=execution_order,
    )


def _make_model_with_constant_node():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 6, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 6, 6])

    reshape_shape = helper.make_tensor(
        name="reshape_shape_value",
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[1, 1, 6, 6],
    )

    nodes = [
        helper.make_node("Relu", inputs=["x"], outputs=["n0_out"], name="n0"),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["reshape_shape"],
            value=reshape_shape,
            name="shape_const",
        ),
        helper.make_node(
            "Reshape",
            inputs=["n0_out", "reshape_shape"],
            outputs=["n1_out"],
            name="n1",
        ),
        helper.make_node("Relu", inputs=["n1_out"], outputs=["y"], name="n2"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="constant_node_model",
        inputs=[x],
        outputs=[y],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    onnx.checker.check_model(model)
    return model


def test_rewrite_removes_constant_nodes_before_splitting():
    model = _make_model_with_constant_node()
    assert any(node.op_type == "Constant" for node in model.graph.node)

    rewritten = rewrite_model(model, [])

    assert all(node.op_type != "Constant" for node in rewritten.graph.node)
    ok, diffs = verify_model(model, rewritten)
    assert ok, f"rewritten model output mismatch: {diffs}"


def test_rewrite_uses_node_indices_after_constant_removal():
    model = _make_model_with_constant_node()
    group = _make_group(
        node_range=(1, 2),
        tile_count=(1, 1),
        execution_order=[
            (1, (0, 0)),
            (2, (0, 0)),
        ],
    )

    rewritten = rewrite_model(model, [group])

    assert any(node.name == "n2_split_s0_0" for node in rewritten.graph.node)
    ok, diffs = verify_model(model, rewritten)
    assert ok, f"rewritten model output mismatch: {diffs}"


def test_remove_constant_nodes_reports_unremovable_constant():
    bad_const = gs.Node(
        op="Constant",
        name="bad_const",
        attrs={"value_int": 3},
        inputs=[],
        outputs=[gs.Variable("bad_const_out")],
    )
    passthrough_in = gs.Variable("x")
    passthrough_out = gs.Variable("y")
    passthrough = gs.Node(
        op="Relu",
        name="relu",
        inputs=[passthrough_in],
        outputs=[passthrough_out],
    )
    graph = gs.Graph(
        nodes=[bad_const, passthrough],
        inputs=[passthrough_in],
        outputs=[passthrough_out],
    )

    with pytest.raises(AssertionError, match="Failed to remove all Constant nodes"):
        _remove_constant_nodes(graph)


def test_rewrite_hard_fails_if_any_constant_cannot_be_removed():
    model = _make_model_with_constant_node()
    with patch(
        "ts.rewrite.pipeline._remove_constant_nodes",
        side_effect=AssertionError(
            "Failed to remove all Constant nodes before splitting: bad_const: unsupported Constant attrs ['value_int']"
        ),
    ):
        with pytest.raises(AssertionError, match="Failed to remove all Constant nodes"):
            rewrite_model(model, [])
