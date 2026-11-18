"""CIFAR-10 loading and GPU-free, batched data augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def load_split(root: str | Path, split: str, cache_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """Load ``root/{split}/{class}/*.{png,jpg}`` into uint8 [N, 32, 32, 3] plus int64 labels.

    Decoded arrays are cached as .npz because decoding 60k small images dominates
    start-up time otherwise.
    """
    cache = Path(cache_dir) / f"cifar10_{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    from PIL import Image

    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        files = sorted(p for p in (Path(root) / split / name).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not files:
            raise FileNotFoundError(f"no images in {Path(root) / split / name}")
        for f in files:
            xs.append(np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8))
            ys.append(label)
    x, y = np.stack(xs), np.asarray(ys, dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y)
    return x, y


def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    """uint8 NHWC -> normalised float32 NCHW."""
    x = (x_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def augment(x: torch.Tensor, gen: torch.Generator, pad: int = 4, cutout: int = 8) -> torch.Tensor:
    """Random crop (reflect-pad 4), horizontal flip and Cutout on a normalised NCHW batch."""
    n, _, h, w = x.shape
    padded = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    dy = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    dx = torch.randint(0, 2 * pad + 1, (n,), generator=gen)
    out = _crop(padded, dy, dx, h, w)

    flip = torch.rand(n, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)

    if cutout > 0:
        cy = torch.randint(0, h, (n,), generator=gen)
        cx = torch.randint(0, w, (n,), generator=gen)
        yy = torch.arange(h)[None, :, None]
        xx = torch.arange(w)[None, None, :]
        mask = ((yy - cy[:, None, None]).abs() < cutout // 2) & ((xx - cx[:, None, None]).abs() < cutout // 2)
        out = out.masked_fill(mask[:, None], 0.0)
    return out


def _crop(padded: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, h: int, w: int) -> torch.Tensor:
    n, c = padded.shape[:2]
    iy = (dy[:, None] + torch.arange(h)[None])  # [n, h]
    ix = (dx[:, None] + torch.arange(w)[None])  # [n, w]
    out = padded[torch.arange(n)[:, None, None], :, iy[:, :, None], ix[:, None, :]]  # [n, h, w, c]
    return out.permute(0, 3, 1, 2).contiguous()
