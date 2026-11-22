"""Independent SNUNet inference: Open-CD LEVIR C16 by default, explicit legacy C32."""
from __future__ import annotations

import argparse
from copy import deepcopy
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
DEFAULT_PROFILE = "levir_c16"
PROFILE_SETTINGS = {
    "levir_c16": {
        "profile": "levir_c16", "trained_on": "LEVIR-CD", "target_object": "building",
        "base_channel": 16, "weight_file": "snunet_levir_c16.pth",
        "labels": {"0": "unchanged", "1": "building_change"},
        "preprocessing": "PIL RGB float32 0..255 CHW, subtract RGB mean then divide std; native 256 tile, no resizing",
        "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375],
        "input_size": [256, 256], "upstream_test_resize": [1024, 1024],
        "deployment_note": "Native 256 tiles; not the published Open-CD whole-image 1024 test protocol",
    },
    "cdd_c32": {
        "profile": "cdd_c32", "trained_on": "CDD", "target_object": "general_change",
        "base_channel": 32, "weight_file": "snunet_cdd.pth",
        "labels": {"0": "unchanged", "1": "change"},
        "preprocessing": "PIL RGB float32 0..255 CHW, no normalization or resizing",
        "input_size": [256, 256],
        "deployment_note": "Legacy CDD general change, not a building specialist",
    },
}


def get_profile(profile: str = DEFAULT_PROFILE) -> dict:
    if profile not in PROFILE_SETTINGS:
        raise ValueError(f"Unknown SNUNet profile: {profile}; choose {tuple(PROFILE_SETTINGS)}")
    return deepcopy(PROFILE_SETTINGS[profile])


def apply_model_configuration(model: torch.nn.Module, configuration: dict) -> None:
    """Restore parameter-free pooling choices from a full-model checkpoint."""
    pools = configuration.get("attention_pooling")
    if not isinstance(pools, dict) or set(pools) != {"ca", "ca1"}:
        raise ValueError("Expected pooling configuration for ca and ca1")
    constructors = {"AdaptiveAvgPool2d": torch.nn.AdaptiveAvgPool2d,
                    "AdaptiveMaxPool2d": torch.nn.AdaptiveMaxPool2d}
    for name, settings in pools.items():
        if not isinstance(settings, dict) or set(settings) != {"avg_pool", "max_pool"}:
            raise ValueError(f"Invalid pooling configuration: {name}")
        for field, kind in settings.items():
            if kind not in constructors:
                raise ValueError(f"Unsupported pooling type: {kind}")
            setattr(getattr(model, name), field, constructors[kind](1))


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
    """Profile-bound architecture, weights and preprocessing -> uint8 0/1 mask."""

    def __init__(self, weight_path: str | Path | None = None, device: str = "cuda",
                 profile: str = DEFAULT_PROFILE):
        self.profile = profile
        settings = get_profile(profile)
        weight = Path(weight_path or Path(__file__).resolve().parent / "checkpoints" / settings["weight_file"]).expanduser().resolve()
        self.weight_path = weight
        if not weight.is_file():
            raise FileNotFoundError(f"SNUNet checkpoint not found: {weight}; see algorithms/snunet/WEIGHTS.md")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; configure device='cpu' explicitly")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("SNUNet integration supports cpu or cuda devices")
        try:
            checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("Cannot read a tensor-only SNUNet checkpoint; for Open-CD use convert_opencd_checkpoint.py first") from exc
        if profile == "levir_c16":
            from .models.opencd_c16 import SNUNetC16, UPSTREAM_COMMIT as c16_commit
            if not isinstance(checkpoint, dict) or checkpoint.get("profile") != profile or "state_dict" not in checkpoint:
                raise ValueError("C16 requires a converted checkpoint with profile='levir_c16'; CDD C32 cannot be substituted")
            model = SNUNetC16()
            self.upstream_commit = c16_commit
            self.configuration = dict(settings, decoder="FCNHead num_convs=0, two-class logits",
                                      source_sha256=checkpoint.get("source_sha256"),
                                      source_revision=checkpoint.get("source_revision"))
        else:
            if isinstance(checkpoint, dict) and checkpoint.get("profile") not in (None, "cdd_c32"):
                raise ValueError("Checkpoint profile does not match legacy CDD C32")
            model = SNUNet_ECAM(in_ch=3, out_ch=2)
            pools = {"attention_pooling": {
                "ca": {"avg_pool": "AdaptiveAvgPool2d", "max_pool": "AdaptiveMaxPool2d"},
                "ca1": {"avg_pool": "AdaptiveAvgPool2d", "max_pool": "AdaptiveMaxPool2d"},
            }}
            if isinstance(checkpoint, dict) and "model_configuration" in checkpoint:
                pools = checkpoint["model_configuration"]
            elif isinstance(checkpoint, dict) and "source_sha256" in checkpoint:
                raise ValueError("Legacy conversion lacks pooling configuration; rerun convert_checkpoint.py from the original .pt")
            apply_model_configuration(model, pools)
            self.upstream_commit = UPSTREAM_COMMIT
            self.configuration = dict(settings, **pools)
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
    def _preprocess(path: Path, profile: str = DEFAULT_PROFILE) -> torch.Tensor:
        settings = get_profile(profile)
        with Image.open(path) as image:
            array = np.asarray(image, dtype=np.float32).copy()
        if profile == "levir_c16":
            # PIL already supplies RGB: do not perform OpenCV's BGR->RGB again.
            array = (array - np.asarray(settings["mean"], dtype=np.float32)) / np.asarray(settings["std"], dtype=np.float32)
        return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)

    @torch.inference_mode()
    def detect(self, image_before: str, image_after: str) -> np.ndarray:
        before, after = validate_image(image_before), validate_image(image_after)
        a = self._preprocess(before, self.profile).to(self.device)
        b = self._preprocess(after, self.profile).to(self.device)
        prediction = self.model(a, b)
        logits = prediction if self.profile == "levir_c16" else prediction[-1]
        if tuple(logits.shape) != (1, 2, 256, 256) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite 1x2x256x256 logits")
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_before", required=True)
    parser.add_argument("--image_after", required=True)
    parser.add_argument("--weight", help="Defaults to the selected deployment profile's converted weight")
    parser.add_argument("--profile", choices=tuple(PROFILE_SETTINGS), default=DEFAULT_PROFILE)
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
    mask = SNUNet(args.weight, args.device, args.profile).detect(args.image_before, args.image_after)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)
    print(f"mask={output}; changed_pixels={int(mask.sum())}; profile={args.profile}; labels={get_profile(args.profile)['labels']}")


if __name__ == "__main__":
    main()
