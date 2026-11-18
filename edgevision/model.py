"""MobileNetV3-Small adapted for 32x32 CIFAR-10 images.

The block layout and the ImageNet-pretrained checkpoint come from
d-li14/mobilenetv3.pytorch (MIT, see third_party/LICENSE-mobilenetv3.pytorch).
Changes made here for small images and clean deployment:

* the stem and first block run at stride 1 instead of 2 (total stride 8, not 32),
  so a 32x32 input ends with a 4x4 feature map instead of collapsing to 1x1;
* squeeze-excite FC layers are expressed as 1x1 convolutions, so the exported
  ONNX graph is only Conv / HardSwish / HardSigmoid / Mul / Add / pooling / Gemm,
  with no dynamic Shape->Reshape chains;
* hard-swish / hard-sigmoid use the fused PyTorch ops, which export to single
  ONNX nodes.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.hardsigmoid(x)


class HSwish(nn.Module):
    def forward(self, x):
        return F.hardswish(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        hidden = _make_divisible(channel // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channel, 1),
            HSigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


def conv_bn(inp: int, oup: int, k: int, stride: int, act: nn.Module) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, k, stride, (k - 1) // 2, bias=False), nn.BatchNorm2d(oup), act)


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden, oup, k, stride, use_se, use_hs):
        super().__init__()
        self.identity = stride == 1 and inp == oup
        act = HSwish if use_hs else (lambda: nn.ReLU(inplace=True))
        if inp == hidden:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden, hidden, k, stride, (k - 1) // 2, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                act(),
                SqueezeExcite(hidden) if use_se else nn.Identity(),
                nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                act(),
                nn.Conv2d(hidden, hidden, k, stride, (k - 1) // 2, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                SqueezeExcite(hidden) if use_se else nn.Identity(),
                act(),
                nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return x + self.conv(x) if self.identity else self.conv(x)


# k, expansion, out, SE, hard-swish, stride  (MobileNetV3-Small, Howard et al. 2019)
SMALL_CFG = [
    [3, 1, 16, 1, 0, 2],
    [3, 4.5, 24, 0, 0, 2],
    [3, 3.67, 24, 0, 0, 1],
    [5, 4, 40, 1, 1, 2],
    [5, 6, 40, 1, 1, 1],
    [5, 6, 40, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 6, 96, 1, 1, 2],
    [5, 6, 96, 1, 1, 1],
    [5, 6, 96, 1, 1, 1],
]


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int = 10, small_input: bool = True, dropout: float = 0.2):
        super().__init__()
        cfg = [list(c) for c in SMALL_CFG]
        stem_stride = 2
        if small_input:  # keep resolution for 32x32 inputs
            stem_stride = 1
            cfg[0][5] = 1
        inp = 16
        layers: list[nn.Module] = [conv_bn(3, inp, 3, stem_stride, HSwish())]
        for k, t, c, se, hs, s in cfg:
            out = _make_divisible(c, 8)
            hidden = _make_divisible(inp * t, 8)
            layers.append(InvertedResidual(inp, hidden, out, k, s, se, hs))
            inp = out
        self.features = nn.Sequential(*layers)
        self.conv = conv_bn(inp, hidden, 1, 1, HSwish())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1024),
            HSwish(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(self.conv(self.features(x)))
        return self.classifier(torch.flatten(x, 1))


def load_imagenet_backbone(model: MobileNetV3Small, ckpt: str | Path) -> list[str]:
    """Load the ImageNet checkpoint into everything except the final 1000-way layer.

    SE linear weights [out, in] are reshaped to 1x1 conv weights [out, in, 1, 1].
    Returns the list of parameters left at their fresh initialisation.
    """
    src = torch.load(ckpt, map_location="cpu", weights_only=True)
    dst = model.state_dict()
    loaded = {}
    for k, v in src.items():
        if k.startswith("classifier.3."):
            continue  # 1000-class head is replaced by a 10-class one
        if k not in dst:
            raise KeyError(f"unexpected checkpoint key {k}")
        if dst[k].dim() == 4 and v.dim() == 2:
            v = v[:, :, None, None]
        if dst[k].shape != v.shape:
            raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(dst[k].shape)}")
        loaded[k] = v
    model.load_state_dict(loaded, strict=False)
    return sorted(set(dst) - set(loaded))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
