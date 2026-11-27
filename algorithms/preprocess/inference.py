"""Image preprocessing / repair primitives for bi-temporal change detection.

Thin wrappers over battle-tested OpenCV / scikit-image implementations — no
novel algorithms here. Every function takes and returns ``uint8`` RGB HWC
arrays so the repaired images stay compatible with the existing detection tools.

These operations are module 4 (图像预处理与修复) of the seven-module framework.
They fill the gaps documented in ``docs/preprocessing-audit.md``: the six change
detectors only do minimal internal preprocessing and do NOT denoise, deblur, or
align radiometry. These helpers are the "修复" half of the check→repair chain
(module 2 质量检查 finds problems; module 4 attempts to fix them).
"""
from __future__ import annotations

import numpy as np

VERSION = "0.1.0"


def _as_uint8_rgb(image: np.ndarray, name: str) -> np.ndarray:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
        raise ValueError(
            f"{name} must be a uint8 HWC RGB array, got shape="
            f"{getattr(image, 'shape', None)} dtype={getattr(image, 'dtype', None)}")
    return image


def histogram_match(reference: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Match ``source``'s per-channel histogram to ``reference``.

    Radiometric alignment: makes T2's brightness/colour distribution resemble
    T1's, so a change detector sees real change instead of illumination drift.
    Wraps ``skimage.exposure.match_histograms``.
    """
    from skimage.exposure import match_histograms  # optional-heavy lib, import lazily
    reference = _as_uint8_rgb(reference, "reference")
    source = _as_uint8_rgb(source, "source")
    matched = match_histograms(source, reference, channel_axis=-1)
    if matched.dtype != np.uint8:
        matched = np.clip(np.round(matched), 0, 255).astype(np.uint8)
    return matched


def denoise(image: np.ndarray, method: str = "nlmeans", **params) -> np.ndarray:
    """Denoise a uint8 RGB image with a classic OpenCV filter.

    ``method`` ∈ {"nlmeans", "bilateral", "median", "gaussian"}. Extra keyword
    arguments are forwarded to the chosen filter (e.g. ``h`` for nlmeans,
    ``d``/``sigma_color``/``sigma_space`` for bilateral, ``ksize`` for
    median/gaussian).
    """
    import cv2  # heavy C extension, import lazily
    image = _as_uint8_rgb(image, "image")
    if method == "nlmeans":
        h = float(params.get("h", 10))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
        return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    if method == "bilateral":
        return cv2.bilateralFilter(
            image, int(params.get("d", 9)),
            float(params.get("sigma_color", 75)), float(params.get("sigma_space", 75)))
    if method == "median":
        ksize = int(params.get("ksize", 5))
        return cv2.medianBlur(image, ksize + 1 if ksize % 2 == 0 else ksize)
    if method == "gaussian":
        ksize = int(params.get("ksize", 5))
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    raise ValueError(f"unknown denoise method: {method}")


def deblur(image: np.ndarray, method: str = "unsharp", **params) -> np.ndarray:
    """Sharpen a uint8 RGB image with an unsharp mask (a lightweight deblur proxy).

    True blind deblur needs a point-spread function; unsharp masking is the
    standard sharpening stand-in. ``amount`` controls strength, ``sigma`` the
    blur scale subtracted from the original.
    """
    import cv2
    image = _as_uint8_rgb(image, "image")
    if method != "unsharp":
        raise ValueError(f"unknown deblur method: {method}")
    sigma = float(params.get("sigma", 1.0))
    amount = float(params.get("amount", 1.5))
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharp = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
