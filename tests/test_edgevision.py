import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from edgevision.data import augment, to_tensor
from edgevision.model import MobileNetV3Small, load_imagenet_backbone

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "weights" / "mobilenetv3-small-55df8e1f.pth"


def test_output_shape_and_resolution():
    m = MobileNetV3Small(num_classes=10).eval()
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    # Small-input variant keeps a 4x4 map before global pooling (total stride 8).
    assert m.conv(m.features(torch.randn(1, 3, 32, 32))).shape[-2:] == (4, 4)


@pytest.mark.skipif(not CKPT.exists(), reason="pretrained checkpoint not present")
def test_imagenet_backbone_loads_everything_but_the_head():
    m = MobileNetV3Small(num_classes=10)
    fresh = load_imagenet_backbone(m, CKPT)
    assert fresh == ["classifier.3.bias", "classifier.3.weight"]


def test_augment_preserves_shape_and_is_seeded():
    x = to_tensor(np.random.default_rng(0).integers(0, 255, (16, 32, 32, 3), dtype=np.uint8))
    a = augment(x, torch.Generator().manual_seed(1))
    b = augment(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape
    assert torch.equal(a, b)
    assert not torch.equal(a, x)


def test_augment_crop_is_a_shift_of_the_original():
    x = torch.arange(32 * 32, dtype=torch.float32).view(1, 1, 32, 32)
    out = augment(x, torch.Generator().manual_seed(3), cutout=0)
    # Away from the reflected border, every row is a contiguous run of the original row values.
    row = out[0, 0, 16]
    diffs = row[5:27].diff().abs()
    assert torch.all(diffs == 1)


def test_onnx_export_and_int8_quantization(tmp_path):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    torch.manual_seed(0)
    m = MobileNetV3Small(num_classes=10).eval()
    fp32 = tmp_path / "m.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(m, (torch.randn(1, 3, 32, 32),), str(fp32), dynamo=False, opset_version=17,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    ops = {n.op_type for n in onnx.load(fp32).graph.node}
    assert {"Conv", "HardSwish", "HardSigmoid", "Mul", "GlobalAveragePool", "Gemm"} <= ops
    assert not ops & {"Shape", "Gather", "BatchNormalization"}  # static graph, BN folded

    x = torch.randn(4, 3, 32, 32)
    s = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        np.testing.assert_allclose(s.run(None, {"input": x.numpy()})[0], m(x).numpy(), atol=1e-4)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": torch.randn(8, 3, 32, 32).numpy()} for _ in range(2)])

        def get_next(self):
            return next(self.it, None)

    int8 = tmp_path / "q.onnx"
    quantize_static(str(fp32), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    assert int8.stat().st_size < fp32.stat().st_size / 2.5
    q = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    assert q.run(None, {"input": x.numpy()})[0].shape == (4, 10)
