"""Preprocessing / repair tools (module 4): histogram_match, denoise, deblur.

Wrap ``algorithms/preprocess/inference.py`` as lagent ``BaseAction`` tools,
following ``docs/tool-contract.md`` with the same envelope as the quality/registration tools.
These tools MODIFY images (unlike the diagnostic review tools), so they write a
repaired pair plus a ``pair.json`` for the downstream detector.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode

from algorithms.preprocess.inference import VERSION
from algorithms.preprocess.inference import deblur as _deblur
from algorithms.preprocess.inference import denoise as _denoise
from algorithms.preprocess.inference import histogram_match as _histogram_match
from lagent.actions.parser import JsonParser, ParseError

ROOT = Path(__file__).resolve().parents[1]

CAPABILITY = dict(
    output_encoding="rgb_image",
    target_object="image_preprocessing",
    trained_on="not_applicable",
    input_size=256,
    method_family="classical_image_processing",
)

_OUTPUT_NAMES = {
    "histogram_match": {"before": None, "after": "image_after_matched.png"},
    "denoise": {"before": "image_before_denoised.png", "after": "image_after_denoised.png"},
    "deblur": {"before": "image_before_deblurred.png", "after": "image_after_deblurred.png"},
}

_WARNINGS = {
    "histogram_match": ["Histogram matching aligns global radiometry and can attenuate genuine brightness changes; review before trusting the result."],
    "denoise": ["Denoising can smooth small genuine changes; strength is heuristic and uncalibrated."],
    "deblur": ["Unsharp masking is a sharpening proxy, not true blind deblur (which needs a PSF)."],
}


# Standalone subset of the contract helpers from PR #7 (b27f97f).
class ReviewJsonParser(JsonParser):
    """Object-only JSON for these tools; keep legacy Interpreter parsing untouched."""
    def parse_inputs(self, inputs, name='run'):
        if name not in self._api2param:
            raise ParseError('Undeclared review-tool API')
        if isinstance(inputs, str):
            encoded = inputs.strip()
            if encoded.startswith('```json\n') and encoded.endswith('```'):
                encoded = encoded[8:-3].strip()
            try:
                inputs = json.loads(encoded)
            except (ValueError, TypeError) as error:
                raise ParseError('Review tools require a JSON object') from error
        if not isinstance(inputs, dict):
            raise ParseError('Review tools require a JSON object')
        return super().parse_inputs(inputs, name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Every invocation owns a fresh directory, so existing records are not replaced.
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def description(tool, name, text):
    value = deepcopy(tool.__tool_description__)
    value['name'] = name
    value['description'] = text
    return value


def _arguments(image_before, image_after, output_dir):
    values = {"image_before": image_before, "image_after": image_after, "output_dir": output_dir}
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(name + " must be a nonempty path string")
    before = Path(image_before).expanduser().resolve()
    after = Path(image_after).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for name, path in (("image_before", before), ("image_after", after)):
        if not path.is_file():
            raise ValueError(name + " file does not exist: " + str(path))
    if output.exists() and not output.is_dir():
        raise ValueError("output_dir must be a directory, not a file")
    return before, after, output


def _load_rgb256(path):
    with Image.open(path) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Only PNG/JPEG images are supported: " + str(path))
        if image.mode != "RGB" or image.size != (256, 256):
            raise ValueError(
                f"Expected RGB 256x256, got {image.mode} {image.size}: {path}")
        image.load()  # PNG metadata can appear after the image data.
        if any("nodata" in key.lower().replace("_", "") for key in image.info):
            raise ValueError(f"Explicit NoData metadata in {path}. Repair requires a fully valid tile.")
        if "transparency" in image.info:
            raise ValueError(f"Transparency metadata in {path}. Repair requires a fully valid tile.")
        return np.asarray(image, dtype=np.uint8).copy()


def _artifact(path, kind, reference=None, **fields):
    result = dict(kind=kind, path=str(path.resolve()), sha256=sha256(path), **fields)
    if reference is not None:
        result.update(reference_image=str(reference), size=[256, 256],
                      coordinate_system="pixel_xy_top_left")
    return result


def _pair_mean_abs_diff(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def _run_repair(tool, operation, image_before, image_after, output_dir, **opts):
    try:
        before, after, output = _arguments(image_before, image_after, output_dir)
    except (OSError, ValueError, TypeError) as error:
        return ActionReturn(type=tool.name, state=ActionStatusCode.ARGS_ERROR, errmsg=str(error))

    started = time.perf_counter()
    try:
        hashes = {"image_before": sha256(before), "image_after": sha256(after)}
        arr_before = _load_rgb256(before)
        arr_after = _load_rgb256(after)
        if operation == "histogram_match":
            repaired_before, repaired_after = arr_before, _histogram_match(arr_before, arr_after)
        elif operation == "denoise":
            repaired_before, repaired_after = _denoise(arr_before, **opts), _denoise(arr_after, **opts)
        elif operation == "deblur":
            repaired_before, repaired_after = _deblur(arr_before, **opts), _deblur(arr_after, **opts)
        else:
            raise RuntimeError("unsupported repair operation: " + operation)
    except ValueError as error:
        return ActionReturn(type=tool.name, state=ActionStatusCode.ARGS_ERROR, errmsg=str(error))
    except Exception as error:
        return ActionReturn(type=tool.name, state=ActionStatusCode.API_ERROR, errmsg=str(error))

    try:
        if any(sha256(path) != hashes[name] for name, path in
               (("image_before", before), ("image_after", after))):
            raise RuntimeError("input changed during execution; refusing inconsistent provenance")

        run_id = uuid.uuid4().hex
        run_dir = output / (tool.name + "_" + run_id)
        run_dir.mkdir(parents=True, exist_ok=False)

        names = _OUTPUT_NAMES[operation]
        artifacts = []
        before_path = None
        if names["before"] is not None:
            before_path = run_dir / names["before"]
            Image.fromarray(repaired_before).save(before_path)
            artifacts.append(_artifact(before_path, "repaired_image_before", before))
        after_path = run_dir / names["after"]
        Image.fromarray(repaired_after).save(after_path)
        artifacts.append(_artifact(after_path, "repaired_image_after", after))

        pair_path = run_dir / "pair.json"
        write_json(pair_path, dict(
            image_before=str(before_path or before), image_after=str(after_path),
            reference_image=str(before), shape=[256, 256],
            coordinate_system="pixel_xy_top_left", operation=operation, params=opts))
        artifacts.append(_artifact(pair_path, "repaired_pair"))

        if operation == "histogram_match":
            diagnostics = dict(
                operation=operation, method="skimage.exposure.match_histograms",
                pair_mean_abs_diff_before=round(_pair_mean_abs_diff(arr_before, arr_after), 3),
                pair_mean_abs_diff_after=round(_pair_mean_abs_diff(repaired_before, repaired_after), 3))
            summary = "Histogram-matched T2 onto T1; radiometric alignment applied."
        else:
            diagnostics = dict(operation=operation, params=opts)
            summary = operation + " applied to both images; params=" + str(opts)

        runtime = {"numpy": np.__version__}
        for module, key in (("cv2", "opencv"), ("skimage", "scikit_image")):
            try:
                runtime[key] = __import__(module).__version__
            except ImportError:
                pass
        sources = ["tools/preprocess_tool.py", "algorithms/preprocess/inference.py"]

        result = dict(
            schema_version="0.1", run_id=run_id, summary=summary,
            artifacts=artifacts, diagnostics=diagnostics, warnings=_WARNINGS[operation],
            elapsed_seconds=round(time.perf_counter() - started, 4),
            provenance=dict(
                algorithm=tool.algorithm_name, algorithm_version=VERSION,
                weights="not_applicable", capability=tool.capability, configuration=opts,
                inputs={"image_before": str(before), "image_after": str(after)},
                input_sha256=hashes,
                source_sha256={name: sha256(ROOT / name) for name in sources},
                runtime=runtime, original_inputs_modified=False,
                coordinate_system="raw pixel_xy_top_left; no georeferencing"))
        record_path = run_dir / "result.json"
        result["artifacts"].append(dict(kind="tool_record", path=str(record_path)))
        write_json(record_path, result)
        return result
    except Exception as error:
        return ActionReturn(type=tool.name, state=ActionStatusCode.API_ERROR, errmsg=str(error))


class HistogramMatchTool(BaseAction):
    """Radiometric alignment: match T2's histogram to T1 before change detection."""
    __tool_name__ = "histogram_match"
    source_file = __file__
    algorithm_name = "scikit-image histogram matching"
    capability = CAPABILITY

    def __init__(self):
        super().__init__(parser=ReviewJsonParser, description=description(
            self, "histogram_match",
            "Match the per-channel histogram of T2 (image_after) onto T1 (image_before), "
            "removing global illumination/season drift before change detection. "
            "Outputs the matched T2 plus a repaired pair.json; T1 is left unchanged. "
            "Writes repaired images and a pair only, never detects changes. CPU, no weights. "
            "CAUTION: this GLOBAL match erased genuine building-change signal on LEVIR "
            "(ChangeFormer F1 0.99->0.12 on an easy sample); use it only for confirmed "
            "illumination drift, not as a default preprocessing step for building change."))

    def run(self, image_before: str, image_after: str, output_dir: str = "output"):
        """Match T2's histogram to T1 (radiometric alignment).

        Args:
            image_before (str): Original RGB8 256x256 PNG/JPEG at T1 (reference).
            image_after (str): Original RGB8 256x256 PNG/JPEG at T2 (to be matched).
            output_dir (str): Directory for a unique record and repaired pair.

        Returns:
            dict: Repaired pair and measurements, or an explicit action error.
        """
        return _run_repair(self, "histogram_match", image_before, image_after, output_dir)


class DenoiseTool(BaseAction):
    """Denoise both images with a classic OpenCV filter."""
    __tool_name__ = "denoise"
    source_file = __file__
    algorithm_name = "OpenCV denoising filters"
    capability = CAPABILITY

    def __init__(self):
        super().__init__(parser=ReviewJsonParser, description=description(
            self, "denoise",
            "Denoise both images with a classic OpenCV filter "
            "(nlmeans / bilateral / median / gaussian). Outputs a repaired pair.json. "
            "Writes repaired images and a pair only, never detects changes. CPU, no weights."))

    def run(self, image_before: str, image_after: str, output_dir: str = "output",
            method: str = "nlmeans"):
        """Denoise both images with the chosen filter.

        Args:
            image_before (str): Original RGB8 256x256 PNG/JPEG at T1.
            image_after (str): Original RGB8 256x256 PNG/JPEG at T2.
            output_dir (str): Directory for a unique record and repaired pair.
            method (str): One of nlmeans / bilateral / median / gaussian.

        Returns:
            dict: Repaired pair and parameters, or an explicit action error.
        """
        return _run_repair(self, "denoise", image_before, image_after, output_dir, method=method)


class DeblurTool(BaseAction):
    """Sharpen both images with an unsharp mask (a lightweight deblur proxy)."""
    __tool_name__ = "deblur"
    source_file = __file__
    algorithm_name = "OpenCV unsharp mask"
    capability = CAPABILITY

    def __init__(self):
        super().__init__(parser=ReviewJsonParser, description=description(
            self, "deblur",
            "Sharpen both images with an unsharp mask (a lightweight deblur proxy). "
            "Outputs a repaired pair.json. Writes repaired images and a pair only, "
            "never detects changes. CPU, no weights."))

    def run(self, image_before: str, image_after: str, output_dir: str = "output"):
        """Sharpen both images with an unsharp mask.

        Args:
            image_before (str): Original RGB8 256x256 PNG/JPEG at T1.
            image_after (str): Original RGB8 256x256 PNG/JPEG at T2.
            output_dir (str): Directory for a unique record and repaired pair.

        Returns:
            dict: Repaired pair and parameters, or an explicit action error.
        """
        return _run_repair(self, "deblur", image_before, image_after, output_dir)
