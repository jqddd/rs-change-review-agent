"""Independent DDPM-CD inference using the official LEVIR three-timestep profile."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "algorithms.ddpm_cd"

from .models.sr3_modules.unet import UNet
from .models.sr3_modules.diffusion import GaussianDiffusion
from .models.cd_modules.cd_head_v2 import cd_head_v2

HERE = Path(__file__).resolve().parent
UPSTREAM_COMMIT = "4970792f65227958ffaa1de787649ce2c5839f12"
PROFILE = json.loads((HERE / "config/levir_50_100_400.json").read_text(encoding="utf-8"))
_RNG_LOCK = threading.RLock()


def validate_image(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Image path must be a non-empty string")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Image file does not exist: {resolved}")
    try:
        with Image.open(resolved) as image:
            if image.format not in {"PNG", "JPEG"} or image.mode != "RGB" or image.size != (256, 256):
                raise ValueError(f"Expected aligned RGB 256x256 PNG/JPEG, got {image.format} {image.mode} {image.size}")
            image.load()
    except OSError as exc:
        raise ValueError(f"Cannot read image: {resolved}: {exc}") from exc
    return resolved


@contextmanager
def fixed_noise(seed: int, device: torch.device):
    """Preserve official independent A/B noise draws, with a reproducible seed.

    Restore NumPy and Torch RNG states after execution. Serializes this module's
    own calls; unrelated threads drawing from global RNGs must be coordinated by
    the caller. No claim of bitwise identity across hardware/software versions.
    """
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with _RNG_LOCK, torch.random.fork_rng(devices=devices):
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
        finally:
            np.random.set_state(state)


def make_diffusion(profile: dict) -> GaussianDiffusion:
    return GaussianDiffusion(UNet(**profile["unet"]), image_size=profile["image_size"],
                             channels=3, loss_type="l2", conditional=False)


def make_head(profile: dict) -> cd_head_v2:
    return cd_head_v2(list(profile["feature_scales"]), out_channels=2,
                      inner_channel=profile["unet"]["inner_channel"],
                      channel_multiplier=list(profile["unet"]["channel_mults"]),
                      img_size=profile["image_size"], time_steps=list(profile["timesteps"]))


def load_state(path: Path, model: torch.nn.Module) -> None:
    # Release weights are tensor dictionaries, not full executable model pickles.
    # mmap avoids duplicating a 1.56GB CPU state when the release uses ZIP format.
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(f"Expected a bare tensor state_dict: {path}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)


class DDPMCD:
    """Two checkpoints, decoder features at t=50/100/400, binary building mask."""

    def __init__(self, diffusion_weight: str | Path | None = None,
                 change_weight: str | Path | None = None, device: str = "cuda", seed: int = 0):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.seed = seed
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only cpu or cuda is supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; configure device='cpu' explicitly")
        self.diffusion_weight = Path(diffusion_weight or HERE / "checkpoints" / PROFILE["diffusion_file"]).expanduser().resolve()
        self.change_weight = Path(change_weight or HERE / "checkpoints" / PROFILE["change_file"]).expanduser().resolve()
        for path, field in ((self.diffusion_weight, "diffusion"), (self.change_weight, "change")):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {field} checkpoint: {path}; see algorithms/ddpm_cd/WEIGHTS.md")
            if path.name == PROFILE[field + "_file"] and path.stat().st_size != PROFILE[field + "_bytes"]:
                raise ValueError(f"Incomplete/unexpected {field} release file size: {path.stat().st_size}; expected {PROFILE[field+'_bytes']}")
        # Initialize on meta, materialize strict checkpoint tensors on CPU, then
        # transfer to the requested device. No random-weight fallback is allowed.
        with torch.device("meta"):
            self.diffusion = make_diffusion(PROFILE)
            self.head = make_head(PROFILE)
        self.diffusion.set_new_noise_schedule(PROFILE["noise_schedule"], device="cpu")
        load_state(self.diffusion_weight, self.diffusion)
        load_state(self.change_weight, self.head)
        # Match the upstream wrapper's post-load test-schedule reset.
        self.diffusion.set_new_noise_schedule(PROFILE["noise_schedule"], device="cpu")
        self.diffusion.to(self.device).eval()
        self.head.to(self.device).eval()
        self.configuration = dict(profile=PROFILE, seed=seed, dtype="float32", batch_size=1,
                                  noise_policy="NumPy uniform + independent Torch Gaussian draws for each A/B/t; fixed seed with RNG restoration")

    @staticmethod
    def _preprocess(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1)
        return torch.from_numpy(array).unsqueeze(0).div_(255).mul_(2).sub_(1)

    @torch.inference_mode()
    def _predict_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        selected = set(PROFILE["feature_scales"])
        features_a, features_b = [], []
        for timestep in PROFILE["timesteps"]:
            for image, collection in ((a, features_a), (b, features_b)):
                encoder, decoder = self.diffusion.feats(image, timestep)
                if len(decoder) != 15:
                    raise ValueError("Unexpected decoder feature layout")
                collection.append([value if index in selected else None for index, value in enumerate(decoder)])
                del encoder, decoder
        return self.head(features_a, features_b)

    @torch.inference_mode()
    def detect(self, image_before: str, image_after: str) -> np.ndarray:
        before, after = validate_image(image_before), validate_image(image_after)
        a, b = (self._preprocess(path).to(self.device) for path in (before, after))
        with fixed_noise(self.seed, self.device):
            logits = self._predict_logits(a, b)
            mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        if mask.shape != (256, 256):
            raise ValueError(f"Unexpected mask size: {mask.shape}")
        return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_before", required=True)
    parser.add_argument("--image_after", required=True)
    parser.add_argument("--diffusion-weight")
    parser.add_argument("--change-weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or output.suffix.lower() != ".png":
        parser.error("Choose a new .png output filename")
    validate_image(args.image_before); validate_image(args.image_after)
    model = DDPMCD(args.diffusion_weight, args.change_weight, args.device, args.seed)
    mask = model.detect(args.image_before, args.image_after)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)
    print(f"Saved {output}; changed_pixels={int(mask.sum())}; seed={args.seed}")


if __name__ == "__main__":
    main()
