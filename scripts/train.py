"""Fine-tune ImageNet-pretrained MobileNetV3-Small on CIFAR-10.

Usage:
    python scripts/train.py --data /path/to/cifar10 --epochs 10

``--data`` points at a folder with ``train/<class>/*`` and ``test/<class>/*`` images.
Checkpoints are written every epoch, and an interrupted run resumes from the last one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edgevision.data import augment, load_split, to_tensor  # noqa: E402
from edgevision.model import MobileNetV3Small, count_params, load_imagenet_backbone  # noqa: E402


@torch.no_grad()
def evaluate(model, x, y, batch=500):
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(x), batch):
        logits = model(x[i:i + batch])
        loss += F.cross_entropy(logits, y[i:i + batch], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(x), loss / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--pretrained", default="weights/mobilenetv3-small-55df8e1f.pth")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    xtr, ytr = load_split(args.data, "train")
    xte, yte = load_split(args.data, "test")
    xtr_t, ytr_t = to_tensor(xtr), torch.from_numpy(ytr)
    xte_t, yte_t = to_tensor(xte), torch.from_numpy(yte)

    model = MobileNetV3Small(num_classes=10).to(memory_format=torch.channels_last)
    fresh = load_imagenet_backbone(model, args.pretrained)
    print(f"params={count_params(model):,}  freshly initialised: {fresh}")

    steps_per_epoch = math.ceil(len(xtr_t) / args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.15,
                                                div_factor=10, final_div_factor=100)

    start_epoch, ckpt = 0, out / "last.pt"
    log_path = Path("results/train_log.csv")
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch = state["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")
    elif log_path.exists():
        log_path.unlink()

    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        gen.manual_seed(args.seed * 1000 + epoch)  # deterministic per epoch, so resume is exact
        perm = torch.randperm(len(xtr_t), generator=gen)
        t0, run_loss, run_correct = time.time(), 0.0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch:(step + 1) * args.batch]
            xb = augment(xtr_t[idx], gen).contiguous(memory_format=torch.channels_last)
            yb = ytr_t[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(1) == yb).sum().item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {loss.item():.3f} "
                      f"lr {sched.get_last_lr()[0]:.4f} {time.time() - t0:.0f}s", flush=True)
        train_time = time.time() - t0
        acc, test_loss = evaluate(model, xte_t, yte_t)
        row = dict(epoch=epoch, train_loss=run_loss / len(xtr_t), train_acc=run_correct / len(xtr_t),
                   test_loss=test_loss, test_acc=acc, epoch_seconds=round(train_time, 1))
        print(json.dumps(row), flush=True)
        log_path.parent.mkdir(exist_ok=True)
        new = not log_path.exists()
        with log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch}, ckpt)

    torch.save(model.state_dict(), out / "edgevision_fp32.pt")
    acc, _ = evaluate(model, xte_t, yte_t)
    Path("results").mkdir(exist_ok=True)
    Path("results/train_summary.json").write_text(json.dumps(
        {"epochs": args.epochs, "test_acc": acc, "params": count_params(model),
         "pretrained": Path(args.pretrained).name}, indent=2))
    print(f"final test accuracy {acc:.4f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
