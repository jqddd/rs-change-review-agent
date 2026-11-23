"""Conservative, weight-free translation checks, independent of Lagent.

T1 (A) is always the reference; T2 (B) is the moving image.  Coordinates are
raw image pixels, x right and y down, with no resizing or EXIF transposition.
For a measured displacement (dx, dy), the sampling matrix maps A(x, y) to
B(x + dx, y + dy).  This is OpenCV ECC's convention and therefore requires
WARP_INVERSE_MAP when producing an image of B on A's grid.

The checks below are conservative internal consistency checks.  Neither a
phase response, ECC objective, nor increased image correlation estimates
independent registration accuracy, TRE, or calibrated confidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


PathLike = Union[str, Path]
VERSION = "0.1.0"
SIZE = 256


@dataclass(frozen=True)
class RegistrationConfig:
    """Deployment configuration; never accept these knobs as tool-call args.

    Only translation is supported.  Distances are pixels, overlap is the
    fraction of the reference grid with fully supported bilinear samples,
    and correlation thresholds are internal Pearson correlation gates.
    ``method='ecc'`` uses phase initialization followed by translation ECC.
    """

    method: str = "phase"
    max_displacement_px: float = 24.0
    min_overlap_fraction: float = 0.80
    noop_displacement_px: float = 0.50
    min_texture_std: float = 0.02
    min_gradient_rms: float = 0.008
    max_periodic_autocorrelation: float = 0.95
    min_phase_response: float = 0.15
    min_correlation: float = 0.65
    min_correlation_gain: float = 0.03
    ecc_max_iterations: int = 200
    ecc_epsilon: float = 1e-6
    ecc_stability_px: float = 0.02
    ecc_max_refinement_px: float = 2.0

    def __post_init__(self):
        if self.method not in ("phase", "ecc"):
            raise ValueError("method must be 'phase' or 'ecc'; only translation is supported")
        numeric = asdict(self)
        numeric.pop("method")
        for name, value in numeric.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("{} must be a finite number".format(name))
            try:
                finite = bool(np.isfinite(float(value)))
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("{} must be a finite number".format(name)) from exc
            if not finite:
                raise ValueError("{} must be a finite number".format(name))
        if not 0 < self.max_displacement_px <= 64:
            raise ValueError("max_displacement_px must be in (0, 64]")
        if not 0 < self.noop_displacement_px < self.max_displacement_px:
            raise ValueError("noop_displacement_px must be positive and below max_displacement_px")
        for name in ("min_overlap_fraction", "min_texture_std", "min_gradient_rms",
                     "max_periodic_autocorrelation", "min_phase_response",
                     "min_correlation", "min_correlation_gain"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError("{} must be in (0, 1]".format(name))
        if not isinstance(self.ecc_max_iterations, int) or not 1 <= self.ecc_max_iterations <= 1000:
            raise ValueError("ecc_max_iterations must be an integer in [1, 1000]")
        if not 0 < self.ecc_epsilon <= 0.01 or not 0 < self.ecc_stability_px <= 0.5:
            raise ValueError("invalid ECC stopping tolerances")
        if not 0 < self.ecc_max_refinement_px <= 8:
            raise ValueError("ecc_max_refinement_px must be in (0, 8]")


def _config(config: Optional[RegistrationConfig], default_method: str) -> RegistrationConfig:
    if config is None:
        return RegistrationConfig(method=default_method)
    if not isinstance(config, RegistrationConfig):
        raise ValueError("config must be a RegistrationConfig instance or None")
    return config


def _read(path: PathLike):
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("image path must be a nonempty string or pathlib.Path")
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("only RGB 256x256 PNG/JPEG inputs are supported; GeoTIFF/NoData are unsupported")
    if not path.is_file():
        raise ValueError("image file does not exist: {}".format(path))
    try:
        with Image.open(path) as image:
            if image.format not in ("PNG", "JPEG") or image.mode != "RGB" or image.size != (SIZE, SIZE):
                raise ValueError("{} must be an RGB 256x256 PNG/JPEG; got {} {} {}".format(
                    path, image.format, image.mode, image.size))
            if image.format == "PNG":
                # Pillow exposes RGB16 PNG as mode RGB after reducing it to
                # eight bits.  Reject that silent conversion at the IHDR.
                with path.open("rb") as stream:
                    header = stream.read(25)
                if len(header) < 25 or header[24] != 8:
                    raise ValueError("PNG must store 8 bits per RGB channel; no implicit bit-depth conversion")
            # PNG text chunks may legally follow IDAT and appear in info only
            # after load; validate complete metadata before accepting pixels.
            image.load()
            if any("nodata" in key.lower().replace("_", "") for key in image.info):
                raise ValueError("explicit NoData metadata is unsupported")
            if "transparency" in image.info:
                raise ValueError("transparent/NoData pixels are unsupported")
            orientation = int(image.getexif().get(274, 1))
            pixels = np.array(image, dtype=np.uint8, copy=True)
            metadata = dict(path=str(path), format=image.format, mode="RGB",
                            shape=[SIZE, SIZE, 3], dtype="uint8",
                            bits_per_channel=8,
                            exif_orientation=orientation, exif_transposed=False)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("cannot decode image: {}".format(path)) from exc
    return pixels, metadata


def _gray(pixels):
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.GaussianBlur(gray, (5, 5), 0.8)


def _texture(gray):
    dy, dx = np.gradient(gray)
    centered = gray.astype(np.float64) - float(gray.mean())
    spectrum = np.fft.fft2(centered)
    autocorrelation = np.fft.ifft2(spectrum * spectrum.conj()).real
    denominator = float(autocorrelation[0, 0])
    if denominator <= 1e-12:
        periodicity = 1.0
    else:
        autocorrelation = np.fft.fftshift(autocorrelation / denominator)
        # Exclude immediate neighbors: ordinary spatial smoothness is not repetition.
        autocorrelation[SIZE // 2 - 4:SIZE // 2 + 5, SIZE // 2 - 4:SIZE // 2 + 5] = -1
        periodicity = float(autocorrelation.max())
    return {"std_0_to_1": float(gray.std()),
            "gradient_rms_0_to_1_per_px": float(np.sqrt(np.mean(dx * dx + dy * dy))),
            "periodic_autocorrelation": periodicity}


def _matrix(dx=0.0, dy=0.0):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)


def _valid_mask(matrix):
    y, x = np.indices((SIZE, SIZE), dtype=np.float64)
    sx = x + float(matrix[0, 2])
    sy = y + float(matrix[1, 2])
    return ((sx >= 0) & (sx <= SIZE - 1) & (sy >= 0) & (sy <= SIZE - 1)).astype(np.uint8)


def _warp(image, matrix):
    return cv2.warpAffine(image, matrix, (SIZE, SIZE),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _correlation(a, b, valid):
    a = a[valid != 0].astype(np.float64)
    b = b[valid != 0].astype(np.float64)
    if a.size < 64:
        return None
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator < 1e-12 else float(np.clip(np.dot(a, b) / denominator, -1, 1))


def _transform(matrix):
    dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
    return {
        "model": "translation", "units": "pixels",
        "reference": "image_before", "moving": "image_after",
        "output_grid": "image_before", "output_shape": [SIZE, SIZE, 3],
        "coordinate_system": "raw pixel centers; origin top-left; x right; y down; no georeferencing",
        "sampling_matrix_reference_to_moving": matrix.tolist(),
        "forward_matrix_moving_to_reference": _matrix(-dx, -dy).tolist(),
        "sampling_equation": "output_B_on_A[y,x] = B[y+dy,x+dx]",
        "opencv_flags": "INTER_LINEAR | WARP_INVERSE_MAP",
        "valid_mask_definition": "1 iff 0 <= x+dx <= 255 and 0 <= y+dy <= 255; 0 otherwise",
    }


def _estimate(image_before, image_after, config):
    before, meta_before = _read(image_before)
    after, meta_after = _read(image_after)
    a, b = _gray(before), _gray(after)
    result = {
        "status": "unknown", "method": config.method, "reason": "",
        "diagnostics": {"reference_texture": _texture(a), "moving_texture": _texture(b)},
        "transform": None,
        "input_metadata": {"image_before": meta_before, "image_after": meta_after},
        "config": asdict(config),
        "warnings": [
            "Internal image-similarity checks are not independent accuracy validation, TRE, or calibrated confidence.",
            "Only global translation is modeled; real temporal changes, rotation, scale and parallax may invalidate the estimate.",
            "No georeferencing or NoData mask is interpreted. Explicit NoData and transparent inputs are unsupported.",
            "Raw stored pixels are used; EXIF orientation is recorded but not applied.",
        ],
    }

    def decline(reason, status="unknown"):
        result.update(status=status, reason=reason)
        return result, before, after, None

    for label in ("reference_texture", "moving_texture"):
        texture = result["diagnostics"][label]
        if texture["std_0_to_1"] < config.min_texture_std or texture["gradient_rms_0_to_1_per_px"] < config.min_gradient_rms:
            return decline("insufficient_texture")
        if texture["periodic_autocorrelation"] >= config.max_periodic_autocorrelation:
            return decline("ambiguous_repeated_texture")

    # phaseCorrelate may window inputs in-place, hence copies protect evaluation pixels.
    window = cv2.createHanningWindow((SIZE, SIZE), cv2.CV_32F)
    try:
        shift, response = cv2.phaseCorrelate(a.copy(), b.copy(), window)
    except cv2.error:
        return decline("phase_estimation_failed")
    if not np.isfinite([*shift, response]).all():
        return decline("nonfinite_phase_estimate")
    dx, dy = map(float, shift)
    diagnostics = result["diagnostics"]
    diagnostics.update(phase_shift_reference_to_moving_px=[dx, dy], phase_response=float(response))
    if response <= config.min_phase_response:
        return decline("weak_or_ambiguous_phase_peak")
    matrix = _matrix(dx, dy)
    if float(np.hypot(dx, dy)) > config.max_displacement_px:
        return decline("displacement_exceeds_limit", "refused")
    if float(_valid_mask(matrix).mean()) < config.min_overlap_fraction:
        return decline("insufficient_overlap", "refused")

    if config.method == "ecc":
        phase_matrix = matrix.copy()
        phase_baseline = _correlation(a, b, _valid_mask(matrix))
        # Do not let an unnecessary optimizer move an already matching pair
        # to accommodate a real changed patch.  Noop always preserves pixels.
        already_aligned = (float(np.hypot(dx, dy)) <= config.noop_displacement_px
                           and phase_baseline is not None and phase_baseline >= config.min_correlation)
        if already_aligned:
            diagnostics["ecc_skipped_reason"] = "phase_estimate_within_noop_tolerance_and_internal_similarity_sufficient"
        previous_score = None
        converged = False
        used = 0
        try:
            while not already_aligned and used < config.ecc_max_iterations:
                iterations = min(20, config.ecc_max_iterations - used)
                old = matrix.copy()
                score, matrix = cv2.findTransformECC(
                    a, b, matrix.copy(), cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, config.ecc_epsilon),
                    None, 5)
                used += iterations
                if not np.isfinite(matrix).all() or not np.isfinite(score):
                    return decline("nonfinite_ecc_estimate")
                if not np.allclose(matrix[:, :2], np.eye(2), atol=1e-7):
                    return decline("unexpected_nontranslation_transform", "refused")
                delta = float(np.linalg.norm(matrix[:, 2] - old[:, 2]))
                diagnostics.update(ecc_objective=float(score), ecc_iteration_budget_used=used,
                                   ecc_last_update_px=delta)
                refinement = float(np.linalg.norm(matrix[:, 2] - phase_matrix[:, 2]))
                diagnostics["ecc_refinement_from_phase_px"] = refinement
                if refinement > config.ecc_max_refinement_px:
                    return decline("ecc_disagrees_with_phase_initialization")
                if previous_score is not None and delta <= config.ecc_stability_px and abs(float(score) - previous_score) <= max(1e-5, config.ecc_epsilon * 10):
                    converged = True
                    break
                previous_score = float(score)
        except cv2.error:
            diagnostics["ecc_converged"] = False
            return decline("ecc_failed_to_converge")
        diagnostics["ecc_converged"] = None if already_aligned else converged
        if not already_aligned and not converged:
            return decline("ecc_stability_not_reached")

    dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
    displacement = float(np.hypot(dx, dy))
    valid = _valid_mask(matrix)
    overlap = float(valid.mean())
    diagnostics.update(estimated_shift_reference_to_moving_px=[dx, dy],
                       estimated_displacement_px=displacement,
                       candidate_valid_overlap_fraction=overlap)
    if displacement > config.max_displacement_px:
        return decline("displacement_exceeds_limit", "refused")
    if overlap < config.min_overlap_fraction:
        return decline("insufficient_overlap", "refused")
    baseline = _correlation(a, b, valid)
    candidate = _correlation(a, _warp(b, matrix), valid)
    diagnostics.update(correlation_before_same_candidate_support=baseline,
                       correlation_after_candidate=candidate,
                       correlation_gain=None if baseline is None or candidate is None else candidate - baseline)
    if baseline is None or candidate is None:
        return decline("undefined_internal_correlation")
    if displacement <= config.noop_displacement_px:
        if baseline < config.min_correlation:
            return decline("small_estimate_but_insufficient_similarity")
        result.update(status="aligned", reason="within_noop_tolerance", transform=_transform(_matrix()))
        diagnostics["output_valid_overlap_fraction"] = 1.0
        return result, before, after, _matrix()
    if candidate < config.min_correlation:
        return decline("insufficient_post_alignment_similarity")
    if candidate - baseline < config.min_correlation_gain:
        return decline("insufficient_internal_correlation_gain")
    result.update(status="misaligned", reason="translation_supported_by_internal_checks", transform=_transform(matrix))
    diagnostics["output_valid_overlap_fraction"] = overlap
    return result, before, after, matrix


class AlignmentInspector:
    """Estimate translation without writing or resampling the original images.

    ``inspect`` accepts paths to strict RGB 256x256 PNG/JPEG files.  Its JSON
    result has status aligned/misaligned/unknown/refused.  Unknown/refused
    means no accepted transform and must never be reported as registration
    success.  Invalid input raises ValueError.
    """

    def __init__(self, config: Optional[RegistrationConfig] = None):
        self.config = _config(config, "phase")

    def inspect(self, image_before: PathLike, image_after: PathLike) -> Dict[str, Any]:
        result, _, _, _ = _estimate(image_before, image_after, self.config)
        return result


class PairRegistrar:
    """Put moving T2 on reference T1's raw pixel grid with one RGB resampling.

    Default method is phase-initialized translation ECC.  Returns the same
    metadata as inspection plus ``aligned_image`` (RGB uint8), ``valid_mask``
    (uint8 labels 0/1), and status registered/noop/unknown/refused.  Arrays
    are None for unknown/refused; noop returns unchanged T2 and an all-one
    mask.  The caller owns derived-file storage; originals are never changed.
    """

    def __init__(self, config: Optional[RegistrationConfig] = None):
        self.config = _config(config, "ecc")

    def register(self, image_before: PathLike, image_after: PathLike) -> Dict[str, Any]:
        result, _, after, matrix = _estimate(image_before, image_after, self.config)
        result.update(aligned_image=None, valid_mask=None)
        if result["status"] == "aligned":
            result.update(status="noop", aligned_image=after.copy(),
                          valid_mask=np.ones((SIZE, SIZE), dtype=np.uint8))
        elif result["status"] == "misaligned":
            result.update(status="registered", aligned_image=_warp(after, matrix),
                          valid_mask=_valid_mask(matrix))
        return result
