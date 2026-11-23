"""Deterministic quality screening without a learned blur or NoData model.

Inputs must already be RGB uint8 256 x 256 PNG/JPEG images. This module does
not resize images, convert image modes, infer NoData, or decide whether a pair
contains a real change. Gradient measurements are evidence for review only:
texture and noise can increase them, and low-texture scenes can reduce them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


ALGORITHM_VERSION = "1.0.0"
CONFIG_VERSION = "uncalibrated_v1"
_SIZE = (256, 256)
_TILE_SIZE = 64
_DEFAULTS = {
    "tile_size": _TILE_SIZE,
    "min_variance": 4.0,
    "min_laplacian_variance": 20.0,
    "min_tenengrad_energy": 100.0,
    "min_mean_brightness": 15.0,
    "max_mean_brightness": 240.0,
    "max_clipped_fraction": 0.20,
    "relative_sharpness_ratio": 0.50,
}
_RANGES = {
    "min_variance": (0.0, 65025.0),
    "min_laplacian_variance": (0.0, 1e8),
    "min_tenengrad_energy": (0.0, 1e8),
    "min_mean_brightness": (0.0, 255.0),
    "max_mean_brightness": (0.0, 255.0),
    "max_clipped_fraction": (0.0, 1.0),
    "relative_sharpness_ratio": (0.0, 1.0),
}


def _configuration(config: Optional[Mapping]) -> Mapping:
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("config must be a mapping or None")
    values = dict(_DEFAULTS)
    if config is not None:
        unknown = set(config) - set(values)
        if unknown:
            raise ValueError("Unknown config keys: {}".format(
                ", ".join(sorted(str(key) for key in unknown))))
        values.update(config)
    tile_size = values["tile_size"]
    if (isinstance(tile_size, (bool, np.bool_))
            or not isinstance(tile_size, Integral)
            or tile_size != _TILE_SIZE):
        raise ValueError("tile_size must be the fixed integer 64")
    values["tile_size"] = int(tile_size)
    for key, (lower, upper) in _RANGES.items():
        value = values[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("{} must be a finite real number".format(key))
        try:
            value = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError("{} must be a finite real number".format(key)) from error
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError("{} must be within [{}, {}]".format(key, lower, upper))
        values[key] = value
    if not 0.0 < values["relative_sharpness_ratio"] < 1.0:
        raise ValueError("relative_sharpness_ratio must be strictly between 0 and 1")
    if values["min_mean_brightness"] >= values["max_mean_brightness"]:
        raise ValueError("min_mean_brightness must be less than max_mean_brightness")
    return MappingProxyType(values)


def _load_rgb(path_value: str, label: str):
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("{} must be a nonempty image path string".format(label))
    path = Path(path_value)
    if not path.is_file():
        raise ValueError("{} is not an existing image file: {}".format(label, path_value))
    try:
        # Inspect the encoded source before Pillow can downcast a 16-bit RGB PNG.
        payload = path.read_bytes()
        with Image.open(path) as image:
            image_format = image.format
            mode = image.mode
            size = image.size
            if image_format not in ("PNG", "JPEG"):
                raise ValueError("{} must contain a PNG or JPEG image".format(label))
            if mode != "RGB":
                raise ValueError("{} must have mode RGB; found {}".format(label, mode))
            if size != _SIZE:
                raise ValueError("{} must be 256 x 256; found {} x {}".format(
                    label, size[0], size[1]))
            if image_format == "PNG":
                if (len(payload) < 26 or payload[:8] != b"\x89PNG\r\n\x1a\n"
                        or payload[12:16] != b"IHDR"):
                    raise ValueError("{} has an invalid PNG header".format(label))
                if payload[24] != 8 or payload[25] != 2:
                    raise ValueError("{} must be an 8-bit RGB PNG".format(label))
            image.load()
            if any("nodata" in key.lower().replace("_", "") for key in image.info):
                raise ValueError("{} has explicit NoData metadata; all pixels must be valid".format(label))
            if "transparency" in image.info:
                raise ValueError("{} has transparent/NoData pixels, which are unsupported".format(label))
            rgb = np.array(image, copy=True)
        # Unchanged decoding independently detects high-bit-depth and alpha data.
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("{} could not be decoded".format(label))
        if decoded.dtype != np.uint8 or decoded.shape != (256, 256, 3):
            raise ValueError("{} must decode as RGB uint8 256 x 256".format(label))
        if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
            raise ValueError("{} must contain RGB uint8 samples".format(label))
    except (OSError, UnidentifiedImageError, SyntaxError) as error:
        raise ValueError("{} is not a readable PNG/JPEG image: {}".format(
            label, path_value)) from error
    validation = {
        "valid": True,
        "format": image_format,
        "mode": mode,
        "size": list(size),
        "width": int(size[0]),
        "height": int(size[1]),
        "dtype": str(rgb.dtype),
        "observed_value_range": [int(rgb.min()), int(rgb.max())],
        "checks": {
            "supported_format": True,
            "rgb_mode": True,
            "required_size": True,
            "uint8_dtype": True,
            "value_range_0_255": True,
        },
    }
    return rgb, validation


class ImageQualityInspector:
    """Measure two images using a frozen, uncalibrated screening profile.

    ``config`` is optional application-side configuration, validated and copied
    once at construction. It is not an ``inspect`` argument or a model-tuned
    decision rule. Changing thresholds does not calibrate the profile.
    """

    def __init__(self, config: Optional[Mapping] = None):
        self.config = _configuration(config)

    def _measure(self, rgb: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        percentiles = np.percentile(gray, [5, 50, 95])
        laplacian = cv2.Laplacian(
            gray, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101)
        sobel_x = cv2.Sobel(
            gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        sobel_y = cv2.Sobel(
            gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        brightness = {
            "mean": float(gray.mean()),
            "p05": float(percentiles[0]),
            "p50": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "variance": float(gray.var()),
        }
        clipping = {
            "channel_zero_fraction": float(np.mean(rgb == 0)),
            "channel_255_fraction": float(np.mean(rgb == 255)),
            "black_pixel_fraction": float(np.mean(np.all(rgb == 0, axis=2))),
            "white_pixel_fraction": float(np.mean(np.all(rgb == 255, axis=2))),
        }
        sharpness = {
            "laplacian_variance": float(laplacian.var()),
            "tenengrad_energy": float(np.mean(sobel_x ** 2 + sobel_y ** 2)),
        }
        flags = []
        # Zero-variance images remain unknown even when a caller sets min_variance=0.
        low_texture = brightness["variance"] <= self.config["min_variance"]
        if low_texture:
            flags.append("low_texture_or_constant")
        if brightness["mean"] < self.config["min_mean_brightness"]:
            flags.append("low_mean_brightness")
        if brightness["mean"] > self.config["max_mean_brightness"]:
            flags.append("high_mean_brightness")
        if clipping["channel_zero_fraction"] > self.config["max_clipped_fraction"]:
            flags.append("many_zero_channels")
        if clipping["channel_255_fraction"] > self.config["max_clipped_fraction"]:
            flags.append("many_255_channels")
        if sharpness["laplacian_variance"] < self.config["min_laplacian_variance"]:
            flags.append("low_laplacian_variance_screening")
        if sharpness["tenengrad_energy"] < self.config["min_tenengrad_energy"]:
            flags.append("low_tenengrad_energy_screening")
        return {
            "brightness": brightness,
            "clipping": clipping,
            "sharpness": sharpness,
            "texture_assessment": "unknown" if low_texture else "measurable",
            "assessment": "unknown" if low_texture else ("review" if flags else "metrics_only"),
            "heuristic_flags": flags,
        }

    def _image_report(self, rgb: np.ndarray, validation: dict) -> Dict[str, Any]:
        report = self._measure(rgb)
        report["validation"] = validation
        report["tiles"] = []
        height, width = rgb.shape[:2]
        for y0 in range(0, height, _TILE_SIZE):
            for x0 in range(0, width, _TILE_SIZE):
                x1, y1 = min(x0 + _TILE_SIZE, width), min(y0 + _TILE_SIZE, height)
                tile = self._measure(rgb[y0:y1, x0:x1])
                tile["tile_index"] = len(report["tiles"])
                tile["bbox"] = [x0, y0, x1, y1]
                report["tiles"].append(tile)
        return report

    def _compare(self, before: dict, after: dict) -> Dict[str, Any]:
        ratios = {}
        for name in ("laplacian_variance", "tenengrad_energy"):
            baseline = before["sharpness"][name]
            ratios[name + "_after_over_before"] = (
                float(after["sharpness"][name] / baseline) if baseline > 1e-12 else None)
        values = list(ratios.values())
        unknown = (before["texture_assessment"] == "unknown"
                   or after["texture_assessment"] == "unknown"
                   or any(value is None for value in values))
        flags = []
        direction = "mixed_or_similar"
        if unknown:
            direction = "unknown"
            flags.append("insufficient_texture_for_sharpness_interpretation")
        elif all(value < self.config["relative_sharpness_ratio"] for value in values):
            direction = "after_lower"
            flags.append("after_has_lower_gradient_metrics")
        elif all(value > 1.0 / self.config["relative_sharpness_ratio"] for value in values):
            direction = "before_lower"
            flags.append("before_has_lower_gradient_metrics")
        return {
            **ratios,
            "direction": direction,
            "assessment": "unknown" if unknown else ("review" if flags else "metrics_only"),
            "heuristic_flags": flags,
            "interpretation": (
                "Relative gradient measurements are review clues, not a blur diagnosis or quality ranking. "
                "Texture, noise, compression, exposure, and scene content can change these values."),
        }

    def inspect(self, image_before: str, image_after: str) -> Dict[str, Any]:
        """Return JSON-compatible measurements, or raise ValueError for bad input."""
        before_rgb, before_validation = _load_rgb(image_before, "image_before")
        after_rgb, after_validation = _load_rgb(image_after, "image_after")
        if before_rgb.shape != after_rgb.shape:
            raise ValueError("The before and after image dimensions must match")
        before = self._image_report(before_rgb, before_validation)
        after = self._image_report(after_rgb, after_validation)
        comparison = self._compare(before, after)
        flags = (["before." + flag for flag in before["heuristic_flags"]]
                 + ["after." + flag for flag in after["heuristic_flags"]]
                 + ["pair." + flag for flag in comparison["heuristic_flags"]])
        return {
            "algorithm_version": ALGORITHM_VERSION,
            "config_version": CONFIG_VERSION,
            "threshold_profile": CONFIG_VERSION,
            "config": dict(self.config),
            "assessment": "review" if flags else "metrics_only",
            "limits": {
                "required_formats": ["PNG", "JPEG"],
                "required_mode": "RGB",
                "required_size": list(_SIZE),
                "required_dtype": "uint8",
                "pixel_range": [0, 255],
                "bbox_convention": "[x0, y0, x1, y1], exclusive upper bounds",
                "brightness_definition": "BT.601 luma via cv2.COLOR_RGB2GRAY, 0-255 scale",
                "laplacian_definition": "Variance of CV_64F Laplacian, kernel size 1",
                "tenengrad_definition": "Mean squared magnitude of CV_64F Sobel, kernel size 3",
                "gradient_border": "BORDER_REFLECT_101; each tile measured independently",
                "clipping_definition": "Fractions of encoded RGB channel samples at 0 or 255",
                "validity_assumption": "All input pixels are valid; no valid mask is supplied or inferred",
                "configuration": dict(self.config),
            },
            "images": {"before": before, "after": after},
            "pair": {"sharpness_comparison": comparison},
            "heuristic_flags": flags,
            "limitations": [
                "Thresholds are uncalibrated screening settings; no confidence or probability is estimated.",
                "Low gradients can reflect blur or low texture; noise can increase gradient measurements.",
                "Black and white samples remain valid pixels; NoData is not inferred.",
                "The pair is not tested for unchanged content or against a change-detection ground truth.",
            ],
        }
