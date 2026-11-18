"""Post-training static INT8 quantization with ONNX Runtime.

    python scripts/quantize.py --data /path/to/cifar10 --fp32 models/edgevision_fp32.onnx

Weights are quantized per output channel (int8, symmetric); activations per
tensor (uint8) with ranges calibrated on a held-out slice of the *training*
images, so the test set never influences quantization.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgevision.data import load_split, to_tensor  # noqa: E402


class CifarCalibration(CalibrationDataReader):
    def __init__(self, x: np.ndarray, batch: int = 32):
        self.batches = [x[i:i + batch] for i in range(0, len(x), batch)]
        self.it = iter(self.batches)

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"input": b}

    def rewind(self):
        self.it = iter(self.batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fp32", default="models/edgevision_fp32.onnx")
    ap.add_argument("--out", default="models/edgevision_int8.onnx")
    ap.add_argument("--calib", type=int, default=1024, help="number of calibration images")
    ap.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="percentile")
    ap.add_argument("--format", choices=["qdq", "qoperator"], default="qoperator")
    args = ap.parse_args()

    xtr, _ = load_split(args.data, "train")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xtr), size=args.calib, replace=False)
    calib = to_tensor(xtr[idx]).numpy()

    pre = Path(args.fp32).with_suffix(".pre.onnx")
    quant_pre_process(args.fp32, str(pre), skip_symbolic_shape=True)
    method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile}[args.method]
    quantize_static(
        str(pre), args.out, CifarCalibration(calib),
        quant_format=QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        extra_options={"CalibPercentile": 99.999} if args.method == "percentile" else {},
    )
    pre.unlink(missing_ok=True)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.2f} MB) "
          f"from {Path(args.fp32).stat().st_size / 1e6:.2f} MB FP32")


if __name__ == "__main__":
    main()
