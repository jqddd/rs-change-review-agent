"""Independent Lagent tool adapter; no generic inference/models imports."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from algorithms.snunet.inference import SNUNet, DEFAULT_PROFILE, get_profile, validate_image

ALGO_DIR = Path(__file__).resolve().parents[1] / "algorithms" / "snunet"
CAPABILITY = {
    "output_encoding": "binary",
    "target_object": "building",
    "trained_on": "LEVIR-CD",
    "input_size": 256,
    "method_family": "cnn",
}


class SNUNetTool(BaseAction):
    """SNUNet-CD binary change detection using an explicitly configured checkpoint."""

    __tool_name__ = "snunet"

    def __init__(self, weight_path: Optional[str] = None, device: str = "cuda", trained_on: Optional[str] = None,
                 profile: str = DEFAULT_PROFILE):
        settings = get_profile(profile)
        if trained_on is not None and trained_on != settings["trained_on"]:
            raise ValueError("trained_on must match the selected weight/architecture profile")
        description = deepcopy(self.__tool_description__)
        description['description'] = (
            f"SNUNet {profile}, trained on {settings['trained_on']}, target={settings['target_object']}. "
            f"Labels {settings['labels']}. Accepts aligned fully valid RGB 256x256 PNG/JPEG. "
            "No registration or calibrated confidence; no addition/removal direction. "
            "Deployment profile/weights are fixed, not chosen by the LLM. " + settings['deployment_note'])
        super().__init__(description=description)
        self.profile, self.settings = profile, settings
        self.weight_path = Path(weight_path or ALGO_DIR / "checkpoints" / settings["weight_file"]).expanduser().resolve()
        self.device = device
        self.capability = dict(CAPABILITY, trained_on=settings["trained_on"], target_object=settings["target_object"])
        self._model = None
        self._weight_sha256 = None

    def _get_model(self):
        if self._model is None:
            model = SNUNet(self.weight_path, self.device, profile=self.profile)
            digest = hashlib.sha256()
            with self.weight_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self._weight_sha256 = digest.hexdigest()
            self._model = model
        return self._model

    def run(self, image_before: str, image_after: str, output_dir: str = "output") -> ActionReturn | dict:
        """Run SNUNet on aligned RGB 256x256 PNG/JPEG images, returning binary change masks.

        Default deployment is Open-CD LEVIR C16 building change; the explicit
        cdd_c32 profile is legacy general change. Labels and normalization follow
        that fixed profile, with no added/removed distinction. Requires pre-aligned, fully valid tiles; no
        registration check is performed. GPU/CPU inference cost is measured;
        first call also loads weights. Confidence and quality are not estimated.

        Args:
            image_before (str): Path to the aligned RGB 256x256 image at T1.
            image_after (str): Path to the aligned RGB 256x256 image at T2.
            output_dir (str): Writable output directory; a unique subdirectory is created.

        Returns:
            dict: Contract v0.1 result envelope; failures carry ActionReturn status.
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
                raise ValueError("Algorithm returned an invalid 0/1 256x256 mask")
            run_dir = output / f"snunet_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=False)
            label_path, viz_path = run_dir / "mask_label.png", run_dir / "mask_viz.png"
            Image.fromarray(mask.astype(np.uint8)).save(label_path)
            viz = np.zeros((256, 256, 3), dtype=np.uint8)
            viz[mask == 1] = [255, 0, 0]
            Image.fromarray(viz).save(viz_path)
            result = {
                "schema_version": "0.1", "run_id": run_id,
                "summary": f"SNUNet {self.profile} detected {int(mask.sum())} {self.settings['labels']['1']} pixels; not an object count",
                "artifacts": [
                    {"kind": "mask_label", "path": str(label_path), "labels": deepcopy(self.settings["labels"]),
                     "size": [256, 256], "reference_image": str(before), "coordinate_system": "pixel_xy_top_left"},
                    {"kind": "mask_viz", "path": str(viz_path)},
                ],
                "diagnostics": {"changed_pixel_count": int(mask.sum()), "changed_fraction": float(mask.mean())},
                "warnings": [
                    f"Configured training domain: {self.capability['trained_on']}; target-domain quality must be validated (see VALIDATION.md)",
                    f"Configured target is {self.capability['target_object']}; binary output does not distinguish addition/removal",
                    "Registration is a caller precondition; model confidence is not calibrated",
                    "No georeferencing available; areas are pixel counts, not square metres",
                    self.settings["deployment_note"],
                ],
                "elapsed_seconds": time.perf_counter() - start,
                "provenance": {
                    "algorithm": "SNUNet-CD/SNUNet_ECAM", "upstream_commit": getattr(model, 'upstream_commit', None),
                    "profile": self.profile,
                    "weight": str(self.weight_path), "weight_sha256": self._weight_sha256,
                    "capability": self.capability, "device": str(model.device),
                    "inputs": {"image_before": str(before), "image_after": str(after)},
                    "preprocessing": self.settings["preprocessing"],
                    "model_configuration": model.configuration,
                    "includes_model_load": includes_load, "timing_scope": "load_if_needed+inference+mask_writes; excludes result.json write",
                },
            }
            (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            return ActionReturn(type=self.name, state=ActionStatusCode.API_ERROR, errmsg=f"SNUNet execution failed: {exc}")
