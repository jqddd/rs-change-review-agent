"""TinyCD 算法实现（脱离 lagent 可单独运行）。

TinyCD = 轻量级双时相建筑变化检测（CNN，基于 EfficientNet 骨干）。
输入两张 256×256 双时相影像，输出二值变化掩码（0=不变，1=变化）。

单独验证：
    python inference.py --image_before A.png --image_after B.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from models.change_classifier import ChangeClassifier  # noqa: E402

MEAN = [0.485, 0.456, 0.406]  # ImageNet 归一化
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 256


class TinyCD:
    """TinyCD 建筑变化检测（二值）。"""

    def __init__(self, weight_path: str | Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # pretrained=False：checkpoint 会覆盖骨干权重，避免下载 ImageNet
        self.model = ChangeClassifier(pretrained=False)
        # TinyCD 的 checkpoint 是裸 state_dict（不像 ChangeFormer/BIT 用 model_G_state_dict 包裹）
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self._normalize = transforms.Normalize(mean=MEAN, std=STD)

    def _preprocess(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        tensor = transforms.ToTensor()(img)     # (3, H, W)，[0, 1]
        tensor = self._normalize(tensor)        # ImageNet 归一化
        return tensor.unsqueeze(0)              # (1, 3, H, W)

    @torch.no_grad()
    def detect(self, image_before: str, image_after: str) -> np.ndarray:
        """前向推理，返回 (H, W) 二值掩码：0=不变，1=变化。"""
        ref = self._preprocess(image_before).to(self.device)
        test = self._preprocess(image_after).to(self.device)
        prob = self.model(ref, test)            # 输出 sigmoid 概率图（单通道）
        prob = prob.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
        return (prob > 0.5).astype(np.uint8)    # 阈值 0.5 出二值掩码


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TinyCD 单独推理验证")
    parser.add_argument("--image_before", required=True, help="前时相影像 T1")
    parser.add_argument("--image_after", required=True, help="后时相影像 T2")
    parser.add_argument("--weight", default="pretrained_models/levir_best.pth")
    args = parser.parse_args()

    model = TinyCD(args.weight)
    mask = model.detect(args.image_before, args.image_after)
    print("变化像素数:", int((mask == 1).sum()), "/ 总像素:", mask.size)
