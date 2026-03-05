import argparse
import onnx

ap = argparse.ArgumentParser()
ap.add_argument("model", help="path to .onnx")
args = ap.parse_args()

m = onnx.load(args.model)

for idx, node in enumerate(m.graph.node):
    name = node.name if node.name else "<unnamed>"
    print(f"{idx:4d}\t{node.op_type}\t{name}")
