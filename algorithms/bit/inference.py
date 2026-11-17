"""BIT 算法实现（脱离 lagent 可单独运行）。

BIT = Bi-temporal Image Transformer，双时相建筑变化检测（二值）。
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
from torchvision.transforms import functional as TF

# 让 `from models.networks import ...` 找得到（models 是 vendor 进来的 BIT 包）
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from models.networks import BASE_Transformer  # noqa: E402

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
IMG_SIZE = 256


class BIT:
    """BIT 建筑变化检测（二值）。"""

    def __init__(self, weight_path: str | Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # 对应原仓库 net_G = 'base_transformer_pos_s4_dd8_dedim8'
        self.model = BASE_Transformer(
            input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
            with_pos="learned", enc_depth=1, dec_depth=8, decoder_dim_head=8,
        )
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_G_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        tensor = TF.to_tensor(img)                        # (3, H, W)，[0, 1]
        tensor = TF.normalize(tensor, mean=MEAN, std=STD)  # 归一化到 [-1, 1]
        return tensor.unsqueeze(0)                        # (1, 3, H, W)

    @torch.no_grad()
    def detect(self, image_before: str, image_after: str) -> np.ndarray:
        """前向推理，返回 (H, W) 二值掩码：0=不变，1=变化。"""
        a = self._preprocess(image_before).to(self.device)
        b = self._preprocess(image_after).to(self.device)
        logits = self.model(a, b)                          # 注意：BIT 直接返回 logits（无 [-1]）
        mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        return mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIT 单独推理验证")
    parser.add_argument("--image_before", required=True, help="前时相影像 T1")
    parser.add_argument("--image_after", required=True, help="后时相影像 T2")
    parser.add_argument("--weight", default="checkpoints/best_ckpt.pt")
    args = parser.parse_args()

    model = BIT(args.weight)
    mask = model.detect(args.image_before, args.image_after)
    print("变化像素数:", int((mask == 1).sum()), "/ 总像素:", mask.size)
