"""Compare FP32 and INT8 EdgeVision models: size, accuracy and CPU latency.

    python scripts/benchmark.py --data /path/to/cifar10

Accuracy uses all 10,000 CIFAR-10 test images. Latency is single-image
(batch 1) median / p90 over --iters runs after warm-up, plus batch-64
throughput, at each requested ONNX Runtime thread count.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgevision.data import load_split, to_tensor  # noqa: E402
from edgevision.model import MobileNetV3Small  # noqa: E402


def lat(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[int(0.9 * (len(ts) - 1))]


def ort_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def predict_all(run, x, batch=250):
    return np.concatenate([run(x[i:i + batch]).argmax(1) for i in range(0, len(x), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="models/edgevision_fp32.pt")
    ap.add_argument("--fp32", default="models/edgevision_fp32.onnx")
    ap.add_argument("--int8", default="models/edgevision_int8.onnx")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tinyinfer", default=None, help="path to TinyInfer's python/ dir to add it as an engine")
    args = ap.parse_args()

    xte, yte = load_split(args.data, "test")
    x = to_tensor(xte).numpy()
    one, b64 = x[:1], x[:64]

    model = MobileNetV3Small(num_classes=10)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    engines = {}
    engines["PyTorch FP32 (eager)"] = (Path(args.weights), lambda t: _torch_runner(model, t))
    engines["ONNX Runtime FP32"] = (Path(args.fp32), lambda t: _ort_runner(args.fp32, t))
    engines["ONNX Runtime INT8"] = (Path(args.int8), lambda t: _ort_runner(args.int8, t))
    if args.tinyinfer:
        sys.path.insert(0, args.tinyinfer)
        import tinyinfer

        def ti_runner(t):
            tinyinfer.set_num_threads(t)
            s = tinyinfer.Session(args.fp32)
            return lambda a: s.run(a)[0]
        engines["TinyInfer FP32 (C++)"] = (Path(args.fp32), ti_runner)

    rows, preds = [], {}
    for name, (path, make) in engines.items():
        run = make(max(args.threads))
        p = predict_all(run, x)
        preds[name] = p
        acc = float((p == yte).mean())
        for t in args.threads:
            run = make(t)
            med, p90 = lat(lambda: run(one), args.iters)
            bmed, _ = lat(lambda: run(b64), max(20, args.iters // 10), warmup=3)
            rows.append({"engine": name, "size_mb": path.stat().st_size / 1e6, "accuracy": acc, "threads": t,
                         "latency_ms_median": med, "latency_ms_p90": p90, "throughput_img_s": 64 / (bmed / 1e3)})
            print(json.dumps(rows[-1]), flush=True)

    base = preds["ONNX Runtime FP32"]
    agree = {k: float((v == base).mean()) for k, v in preds.items()}
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps({"rows": rows, "agreement_with_fp32": agree,
                                                    "cpu": platform.processor() or platform.machine(),
                                                    "onnxruntime": ort.__version__,
                                                    "torch": torch.__version__}, indent=2))
    lines = ["| engine | model size | top-1 acc | threads | batch-1 median ms | batch-1 p90 ms | batch-64 img/s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['engine']} | {r['size_mb']:.2f} MB | {100 * r['accuracy']:.2f}% | {r['threads']} | "
                     f"{r['latency_ms_median']:.3f} | {r['latency_ms_p90']:.3f} | {r['throughput_img_s']:.0f} |")
    lines += ["", "Prediction agreement with ONNX Runtime FP32: " +
              ", ".join(f"{k} {100 * v:.2f}%" for k, v in agree.items())]
    (out / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _torch_runner(model, threads):
    torch.set_num_threads(threads)

    def run(a):
        with torch.no_grad():
            return model(torch.from_numpy(a)).numpy()
    return run


def _ort_runner(path, threads):
    s = ort_session(path, threads)
    return lambda a: s.run(None, {"input": a})[0]


if __name__ == "__main__":
    main()
