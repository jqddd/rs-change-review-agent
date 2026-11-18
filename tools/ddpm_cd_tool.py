"""Lagent adapter for the independent DDPM-CD algorithm."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image
from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from algorithms.ddpm_cd.inference import DDPMCD, PROFILE, UPSTREAM_COMMIT, validate_image

CAPABILITY = {
    "output_encoding": "binary",
    "target_object": "building",
    "trained_on": "LEVIR-CD",
    "input_size": 256,
    "method_family": "diffusion",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DDPMCDTool(BaseAction):
    """DDPM-CD LEVIR building change detection with two fixed checkpoints."""

    __tool_name__ = "ddpm_cd"

    def __init__(self, diffusion_weight: str | None = None, change_weight: str | None = None,
                 device: str = "cuda", seed: int = 0):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.change_weight = change_weight
        self.device, self.seed = device, seed
        self._model = None
        self._weights = None

    def _get_model(self):
        if self._model is None:
            model = DDPMCD(self.diffusion_weight, self.change_weight, self.device, self.seed)
            weights = {
                "diffusion": {"path": str(model.diffusion_weight), "sha256": file_hash(model.diffusion_weight)},
                "change_head": {"path": str(model.change_weight), "sha256": file_hash(model.change_weight)},
            }
            self._model, self._weights = model, weights
        return self._model

    def run(self, image_before: str, image_after: str, output_dir: str = "output") -> ActionReturn | dict:
        """Detect building changes using DDPM-CD trained on LEVIR-CD.

        Accepts aligned, fully valid RGB 256x256 PNG/JPEG tiles. Outputs binary
        0=unchanged, 1=building change; no road class or added/removed direction.
        Two checkpoints and t=[50,100,400] are fixed by deployment configuration.
        Runs six diffusion feature passes and a change head, not 2000-step image
        synthesis. FP32 memory and latency can be substantial; 8GB feasibility
        must be measured with actual weights. Seed is fixed by the deployer for
        reproducible noisy features. No registration/quality checks or calibrated
        confidence are provided; use this tool only when such prerequisites hold.

        Args:
            image_before (str): Path of aligned RGB 256x256 T1 image.
            image_after (str): Path of aligned RGB 256x256 T2 image.
            output_dir (str): Writable output directory for an independent run folder.

        Returns:
            dict: Contract v0.1 result envelope; failures return ActionReturn.
        """
        try:
            before, after = validate_image(image_before), validate_image(image_after)
            if not isinstance(output_dir, str) or not output_dir.strip():
                raise ValueError("output_dir must be a non-empty string")
            output = Path(output_dir).expanduser().resolve()
            if output.exists() and not output.is_dir():
                raise ValueError("output_dir points to a file")
        except (ValueError, OSError) as exc:
            return ActionReturn(type=self.name, state=ActionStatusCode.ARGS_ERROR, errmsg=str(exc))

        start = time.perf_counter()
        includes_load = self._model is None
        run_id = uuid.uuid4().hex
        try:
            model = self._get_model()
            mask = model.detect(str(before), str(after))
            if mask.shape != (256, 256) or not np.isin(mask, (0, 1)).all():
                raise ValueError("Algorithm returned an invalid binary 256x256 mask")
            run_dir = output / f"ddpm_cd_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=False)
            label_path, viz_path = run_dir / "mask_label.png", run_dir / "mask_viz.png"
            Image.fromarray(mask.astype(np.uint8)).save(label_path)
            viz = np.zeros((256, 256, 3), dtype=np.uint8)
            viz[mask == 1] = [255, 0, 0]
            Image.fromarray(viz).save(viz_path)
            result = {
                "schema_version": "0.1", "run_id": run_id,
                "summary": f"DDPM-CD detected {int(mask.sum())} building-change pixels; not a building count",
                "artifacts": [
                    {"kind": "mask_label", "path": str(label_path), "labels": {"0": "unchanged", "1": "building_change"},
                     "size": [256, 256], "reference_image": str(before), "coordinate_system": "pixel_xy_top_left"},
                    {"kind": "mask_viz", "path": str(viz_path)},
                ],
                "diagnostics": {"changed_pixel_count": int(mask.sum()), "changed_fraction": float(mask.mean())},
                "warnings": [
                    "Configured weights are LEVIR-CD; target-domain accuracy and multi-model selection must be evaluated",
                    "Requires pre-aligned fully valid tiles; no road class, direction label or calibrated confidence",
                    "No georeferencing: area is in pixels, not square metres",
                    "Feature extraction is stochastic upstream; this deployment fixes and records its noise seed",
                ],
                "elapsed_seconds": time.perf_counter() - start,
                "provenance": {
                    "algorithm": "DDPM-CD", "upstream_commit": UPSTREAM_COMMIT,
                    "weights": self._weights, "capability": dict(CAPABILITY),
                    "inputs": {"image_before": str(before), "image_after": str(after)},
                    "device": str(model.device), "configuration": model.configuration,
                    "includes_model_load": includes_load,
                    "timing_scope": "load/hash_if_needed+preprocess+inference+mask_writes; excludes result.json write",
                },
            }
            (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            return ActionReturn(type=self.name, state=ActionStatusCode.API_ERROR, errmsg=f"DDPM-CD execution failed: {exc}")
