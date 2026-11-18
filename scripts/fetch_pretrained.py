"""Download the ImageNet-pretrained MobileNetV3-Small checkpoint (67.7% top-1)
published by d-li14/mobilenetv3.pytorch (MIT licence) into weights/."""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/d-li14/mobilenetv3.pytorch/master/pretrained/mobilenetv3-small-55df8e1f.pth"


def main():
    dest = Path("weights") / Path(URL).name
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        print(f"{dest} already present")
        return
    urllib.request.urlretrieve(URL, dest)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
