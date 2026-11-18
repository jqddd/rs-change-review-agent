"""Independent SNUNet-CD inference (no Lagent dependency).

Run from the repository root: python -m algorithms.snunet.inference --help
The official CDD pipeline uses RGB float32 values in [0, 255], without Normalize.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Support the same direct-script convention as the other algorithm integrations,
# while retaining qualified imports so multiple `models` packages can coexist.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "algorithms.snunet"

from .models.Models import SNUNet_ECAM

UPSTREAM_COMMIT = "aed3f1833f3d9d3eccba8c894fd097f358b5329e"
INPUT_SIZE = (256, 256)


def validate_image(path: str) -> Path:
    """Accept only full, aligned RGB PNG/JPEG tiles; never silently resize."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Image path must be a non-empty string")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Image file does not exist: {resolved}")
    try:
        with Image.open(resolved) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("Only PNG/JPEG without geospatial metadata are supported")
            if image.mode != "RGB" or image.size != INPUT_SIZE:
                raise ValueError(f"Expected RGB 256x256, got {image.mode} {image.size}: {resolved}")
            image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot read image: {resolved}: {exc}") from exc
    return resolved


class SNUNet:
    """Official SNUNet_ECAM, two-class logits -> uint8 0/1 change mask."""

    def __init__(self, weight_path: str | Path, device: str = "cuda"):
        weight = Path(weight_path).expanduser().resolve()
        if not weight.is_file():
            raise FileNotFoundError(f"SNUNet checkpoint not found: {weight}; see algorithms/snunet/WEIGHTS.md")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; configure device='cpu' explicitly")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("SNUNet integration supports cpu or cuda devices")
        model = SNUNet_ECAM(in_ch=3, out_ch=2)
        checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
        state = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model_G_state_dict"):
                if key in checkpoint:
                    state = checkpoint[key]
                    break
        if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected a tensor state_dict or a documented state_dict wrapper")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    @staticmethod
    def _preprocess(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            # Exact official ToTensor convention: no /255 and no normalization.
            array = np.asarray(image, dtype=np.float32).copy()
        return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)

    @torch.inference_mode()
    def detect(self, image_before: str, image_after: str) -> np.ndarray:
        before, after = validate_image(image_before), validate_image(image_after)
        a = self._preprocess(before).to(self.device)
        b = self._preprocess(after).to(self.device)
        logits = self.model(a, b)[-1]
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_before", required=True)
    parser.add_argument("--image_after", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, help="New single-channel 0/1 PNG; existing files are refused")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".png":
        parser.error("--output must be a PNG label file")
    if output.exists():
        parser.error("Output already exists; choose a new filename")
    validate_image(args.image_before)
    validate_image(args.image_after)
    mask = SNUNet(args.weight, args.device).detect(args.image_before, args.image_after)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)
    print(f"mask={output}; changed_pixels={int(mask.sum())}; encoding=0:unchanged,1:change")


if __name__ == "__main__":
    main()
