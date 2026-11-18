"""Export the fine-tuned FP32 model to ONNX and check ONNX Runtime against PyTorch.

    python scripts/export_onnx.py --weights models/edgevision_fp32.pt --out models/edgevision_fp32.onnx
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgevision.model import MobileNetV3Small  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/edgevision_fp32.pt")
    ap.add_argument("--out", default="models/edgevision_fp32.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = MobileNetV3Small(num_classes=10)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The TorchScript exporter folds BatchNorm into the preceding Conv in eval mode.
        torch.onnx.export(model, (x,), args.out, dynamo=False, opset_version=args.opset,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})

    xb = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        ref = model(xb).numpy()
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": xb.numpy()})[0]
    diff = float(np.abs(got - ref).max())
    print(f"exported {args.out}: {len(m.graph.node)} nodes, ops={ops}")
    print(f"max |onnxruntime - pytorch| = {diff:.2e}")
    assert diff < 1e-3, "ONNX export does not match PyTorch"


if __name__ == "__main__":
    main()
