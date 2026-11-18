"""Get CIFAR-10 as image folders: data/cifar10/{train,test}/<class>/*.

Uses torchvision's official CIFAR-10 download and writes PNGs, so the training
scripts see the same folder layout regardless of source.
"""

from pathlib import Path


def main(root: str = "data/cifar10"):
    import torchvision

    out = Path(root)
    for split, train in (("train", True), ("test", False)):
        ds = torchvision.datasets.CIFAR10("data/_raw", train=train, download=True)
        for i, (img, label) in enumerate(ds):
            d = out / split / ds.classes[label]
            d.mkdir(parents=True, exist_ok=True)
            img.save(d / f"{i:05d}.png")
        print(f"{split}: {len(ds)} images -> {out / split}")


if __name__ == "__main__":
    main()
