"""Classify image files with an exported EdgeVision ONNX model.

    python scripts/predict.py --model models/edgevision_int8.onnx img1.png img2.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgevision.data import CLASSES, to_tensor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/edgevision_int8.onnx")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    imgs = np.stack([np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR)) for p in args.images])
    logits = sess.run(None, {"input": to_tensor(imgs).numpy()})[0]
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    for path, p in zip(args.images, probs):
        top = np.argsort(-p)[:args.top]
        print(f"{path}: " + ", ".join(f"{CLASSES[i]} {p[i]:.2%}" for i in top))


if __name__ == "__main__":
    main()
