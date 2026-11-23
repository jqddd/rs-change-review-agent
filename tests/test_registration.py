"""Offline CPU tests of geometry and refusal behavior, not field accuracy."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

from algorithms.registration import AlignmentInspector, PairRegistrar, RegistrationConfig


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        # Workspace-local fixtures avoid assumptions about external TEMP permissions.
        self.tmp = tempfile.TemporaryDirectory(
            prefix=".test-registration-", dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.tmp.name)
        self.before = self.root / "A.png"
        self.after = self.root / "B.png"
        rng = np.random.default_rng(8)
        self.reference = cv2.GaussianBlur(
            rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), (5, 5), 0.9)
        self.write(self.reference, self.before)
        self.write(self.reference, self.after)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(array, path):
        Image.fromarray(array).save(path)

    def shifted(self, dx, dy, image=None):
        source = self.reference if image is None else image
        return cv2.warpAffine(source, np.float32([[1, 0, dx], [0, 1, dy]]), (256, 256),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def assert_declined(self, result):
        self.assertIn(result["status"], ("unknown", "refused"))
        self.assertIsNone(result["transform"])
        self.assertIsNone(result["aligned_image"])
        self.assertIsNone(result["valid_mask"])

    def test_defaults_and_json_inspection(self):
        self.assertEqual(AlignmentInspector().config.method, "phase")
        self.assertEqual(PairRegistrar().config.method, "ecc")
        result = AlignmentInspector().inspect(self.before, self.after)
        self.assertEqual(result["status"], "aligned")
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["input_metadata"]["image_before"]["shape"], [256, 256, 3])
        self.assertTrue(Path(result["input_metadata"]["image_before"]["path"]).is_absolute())
        self.assertNotIn("confidence", result["diagnostics"])

    def test_positive_negative_integer_and_subpixel_recovery_direction(self):
        for method in ("phase", "ecc"):
            for dx, dy in ((7, -5), (-9, 6), (2.4, -3.6), (-3.6, 2.4)):
                with self.subTest(method=method, displacement=(dx, dy)):
                    self.write(self.shifted(dx, dy), self.after)
                    originals = [hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (self.before, self.after)]
                    result = PairRegistrar(RegistrationConfig(method=method)).register(self.before, self.after)
                    self.assertEqual(result["status"], "registered", result["reason"])
                    sampling = np.array(result["transform"]["sampling_matrix_reference_to_moving"])
                    forward = np.array(result["transform"]["forward_matrix_moving_to_reference"])
                    np.testing.assert_allclose(sampling[:, 2], [dx, dy], atol=0.3 if method == "phase" else 0.12)
                    np.testing.assert_allclose(forward[:, 2], -sampling[:, 2])
                    np.testing.assert_array_equal(sampling[:, :2], np.eye(2))
                    valid = result["valid_mask"].astype(bool)
                    difference = np.abs(result["aligned_image"].astype(float) - self.reference)[valid]
                    # Fractional fixture creation and correction each interpolate;
                    # restoration cannot recover the original high frequencies.
                    self.assertLess(float(difference.mean()), 8)
                    self.assertEqual(originals, [hashlib.sha256(path.read_bytes()).hexdigest()
                                                  for path in (self.before, self.after)])

    def test_noop_preserves_after_and_has_all_valid_pixels(self):
        for dx, dy in ((0, 0), (0.2, -0.15)):
            moving = self.shifted(dx, dy)
            self.write(moving, self.after)
            result = PairRegistrar().register(self.before, self.after)
            self.assertEqual(result["status"], "noop")
            np.testing.assert_array_equal(result["aligned_image"], moving)
            np.testing.assert_array_equal(result["valid_mask"], np.ones((256, 256), np.uint8))
            np.testing.assert_array_equal(result["transform"]["sampling_matrix_reference_to_moving"],
                                          [[1, 0, 0], [0, 1, 0]])

    def test_constant_and_periodic_texture_are_unknown(self):
        checker = ((np.indices((256, 256)).sum(axis=0) // 8) % 2 * 255).astype(np.uint8)
        for image, reason in ((np.full((256, 256, 3), 120, np.uint8), "insufficient_texture"),
                              (np.repeat(checker[:, :, None], 3, axis=2), "ambiguous_repeated_texture")):
            self.write(image, self.before)
            self.write(image, self.after)
            result = PairRegistrar().register(self.before, self.after)
            self.assert_declined(result)
            self.assertEqual(result["status"], "unknown")
            self.assertEqual(result["reason"], reason)

    def test_excessive_displacement_and_overlap_refused(self):
        self.write(self.shifted(32, 0), self.after)
        result = PairRegistrar().register(self.before, self.after)
        self.assert_declined(result)
        self.assertEqual(result["reason"], "displacement_exceeds_limit")
        self.write(self.shifted(7, -5), self.after)
        result = PairRegistrar(RegistrationConfig(min_overlap_fraction=0.99)).register(self.before, self.after)
        self.assert_declined(result)
        self.assertEqual(result["reason"], "insufficient_overlap")

    def test_ecc_nonconvergence_and_unstable_or_drifting_result_declined(self):
        self.write(self.shifted(7, -5), self.after)
        with patch("algorithms.registration.inference.cv2.findTransformECC", side_effect=cv2.error("not converged")):
            result = PairRegistrar().register(self.before, self.after)
            self.assert_declined(result)
            self.assertEqual(result["reason"], "ecc_failed_to_converge")
        result = PairRegistrar(RegistrationConfig(method="ecc", ecc_max_iterations=1)).register(self.before, self.after)
        self.assert_declined(result)
        self.assertEqual(result["reason"], "ecc_stability_not_reached")

        def drift(a, b, matrix, *args):
            matrix[0, 2] += 4
            return 0.99, matrix

        with patch("algorithms.registration.inference.cv2.findTransformECC", side_effect=drift):
            result = PairRegistrar().register(self.before, self.after)
            self.assert_declined(result)
            self.assertEqual(result["reason"], "ecc_disagrees_with_phase_initialization")

    def test_nonfinite_negative_phase_and_false_internal_improvement_declined(self):
        cases = [(((float("nan"), 0), 0.9), "nonfinite_phase_estimate"),
                 (((0, 0), float("inf")), "nonfinite_phase_estimate"),
                 (((4, 0), -0.8), "weak_or_ambiguous_phase_peak"),
                 (((4, 0), 0.99), "insufficient_post_alignment_similarity")]
        for mocked, reason in cases:
            with self.subTest(reason=reason), patch("algorithms.registration.inference.cv2.phaseCorrelate", return_value=mocked):
                result = PairRegistrar(RegistrationConfig(method="phase")).register(self.before, self.after)
                self.assert_declined(result)
                self.assertEqual(result["reason"], reason)

    def test_valid_border_uses_bilinear_support_and_arrays_have_declared_types(self):
        moving = self.shifted(2.4, -3.6)
        self.write(moving, self.after)
        # Fixed estimator isolates mask geometry from optimizer precision.
        with patch("algorithms.registration.inference.cv2.phaseCorrelate", return_value=((2.4, -3.6), 0.9)):
            result = PairRegistrar(RegistrationConfig(method="phase")).register(self.before, self.after)
        self.assertEqual(result["status"], "registered")
        y, x = np.indices((256, 256))
        expected = ((x + 2.4 <= 255) & (y - 3.6 >= 0)).astype(np.uint8)
        np.testing.assert_array_equal(result["valid_mask"], expected)
        self.assertEqual(result["valid_mask"].dtype, np.uint8)
        self.assertEqual(result["aligned_image"].dtype, np.uint8)
        self.assertEqual(result["aligned_image"].shape, (256, 256, 3))
        np.testing.assert_array_equal(result["valid_mask"][:4], 0)
        np.testing.assert_array_equal(result["valid_mask"][:, -3:], 0)

    def test_real_local_change_is_preserved_or_conservatively_declined(self):
        changed = self.reference.copy()
        changed[90:106, 110:126] = np.minimum(changed[90:106, 110:126].astype(int) + 45, 255)
        self.write(changed, self.after)
        result = PairRegistrar().register(self.before, self.after)
        self.assertEqual(result["status"], "noop", result["reason"])
        np.testing.assert_array_equal(result["aligned_image"], changed)
        self.write(self.shifted(6, -4, changed), self.after)
        result = PairRegistrar().register(self.before, self.after)
        self.assertEqual(result["status"], "registered", result["reason"])
        np.testing.assert_allclose(np.array(result["transform"]["sampling_matrix_reference_to_moving"])[:, 2], [6, -4], atol=0.2)
        difference = result["aligned_image"][93:103, 113:123].astype(float) - self.reference[93:103, 113:123]
        self.assertGreater(float(difference.mean()), 30)
        # A dominant changed patch can destabilize intensity-based ECC: decline it.
        changed[80:130, 100:150] = 255
        self.write(self.shifted(6, -4, changed), self.after)
        self.assert_declined(PairRegistrar().register(self.before, self.after))

    def test_invalid_input_is_not_implicitly_converted_or_resized(self):
        for image in (Image.new("L", (256, 256)), Image.new("RGBA", (256, 256)), Image.new("RGB", (128, 256))):
            image.save(self.after)
            with self.assertRaises(ValueError):
                AlignmentInspector().inspect(self.before, self.after)
        with self.assertRaises(ValueError):
            AlignmentInspector().inspect(self.before, self.root / "missing.png")
        for suffix in (".tif", ".bmp"):
            path = self.root / ("unsupported" + suffix)
            Image.new("RGB", (256, 256)).save(path)
            with self.assertRaises(ValueError):
                AlignmentInspector().inspect(self.before, path)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("NoData", "0")
        Image.fromarray(self.reference).save(self.after, pnginfo=metadata)
        with self.assertRaisesRegex(ValueError, "NoData"):
            PairRegistrar().register(self.before, self.after)
        Image.fromarray(self.reference).save(self.after, transparency=(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "transparent"):
            PairRegistrar().register(self.before, self.after)
        cv2.imwrite(str(self.after), np.full((256, 256, 3), 40000, np.uint16))
        with self.assertRaisesRegex(ValueError, "8 bits"):
            PairRegistrar().register(self.before, self.after)

    def test_exif_is_recorded_and_jpeg_supported_without_transposition(self):
        jpeg = self.root / "B.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(self.reference).save(jpeg, quality=99, subsampling=0, exif=exif)
        result = PairRegistrar().register(jpeg, jpeg)
        self.assertEqual(result["status"], "noop")
        with Image.open(jpeg) as image:
            np.testing.assert_array_equal(result["aligned_image"], np.asarray(image))
        self.assertEqual(result["input_metadata"]["image_after"]["exif_orientation"], 6)
        self.assertFalse(result["input_metadata"]["image_after"]["exif_transposed"])

    def test_nodata_text_after_idat_is_rejected_after_complete_decode(self):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("NoData", "0")
        Image.fromarray(self.reference).save(self.after, pnginfo=metadata)
        original = self.after.read_bytes()
        chunks, cursor = [], 8
        while cursor < len(original):
            length = int.from_bytes(original[cursor:cursor + 4], "big")
            end = cursor + 12 + length
            chunks.append((original[cursor + 4:cursor + 8], original[cursor:end]))
            cursor = end
        # Move complete chunks, including their unchanged length and CRC, so
        # this is a valid late metadata fixture rather than a corrupt PNG.
        text_chunks = [raw for kind, raw in chunks if kind == b"tEXt"]
        self.assertTrue(text_chunks)
        reordered = bytearray(original[:8])
        for kind, raw in chunks:
            if kind == b"tEXt":
                continue
            if kind == b"IEND":
                reordered.extend(b"".join(text_chunks))
            reordered.extend(raw)
        self.after.write_bytes(reordered)
        with Image.open(self.after) as image:
            self.assertNotIn("NoData", image.info)
            image.load()
            self.assertEqual(image.info["NoData"], "0")
            np.testing.assert_array_equal(np.asarray(image), self.reference)
        for operation in (AlignmentInspector().inspect, PairRegistrar().register):
            with self.assertRaisesRegex(ValueError, "NoData"):
                operation(self.before, self.after)

    def test_configuration_rejects_invalid_types_and_values(self):
        for kwargs in ({"method": "affine"}, {"max_displacement_px": float("nan")},
                       {"min_phase_response": -1}, {"ecc_max_iterations": True},
                       {"ecc_max_iterations": 1.5}, {"min_correlation": 2},
                       {"noop_displacement_px": 30}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RegistrationConfig(**kwargs)
        with self.assertRaises(ValueError):
            AlignmentInspector({"method": "phase"})

    def test_configuration_huge_numeric_values_raise_value_error(self):
        for name in ("max_displacement_px", "ecc_max_iterations"):
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "finite number"):
                RegistrationConfig(**{name: 10 ** 999})


if __name__ == "__main__":
    unittest.main()
