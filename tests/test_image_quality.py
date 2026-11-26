"""CPU tests for quality measurements, without a calibrated blur claim."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from algorithms.image_quality import (
    ALGORITHM_VERSION, CONFIG_VERSION, ImageQualityInspector,
)


class ImageQualityInspectorTests(unittest.TestCase):
    def setUp(self):
        fixture_parent = Path(__file__).resolve().parents[1] / "tmp_dir"
        fixture_parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            dir=str(fixture_parent), prefix="test_image_quality_")
        self.root = Path(self.temp.name)
        self.before, self.after = self.root / "A.png", self.root / "B.png"
        rng = np.random.default_rng(17)
        self.texture = rng.integers(30, 226, size=(256, 256, 3), dtype=np.uint8)
        self._save(self.before, self.texture)
        self._save(self.after, self.texture)
        self.inspector = ImageQualityInspector()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _save(path, pixels):
        Image.fromarray(pixels).save(path)

    def inspect(self):
        return self.inspector.inspect(str(self.before), str(self.after))

    def test_json_contract_and_deterministic_finite_measurements(self):
        report = self.inspect()
        self.assertEqual(report, self.inspect())
        self.assertEqual(report, json.loads(json.dumps(report, allow_nan=False)))
        self.assertEqual(report["threshold_profile"], "uncalibrated_v1")
        self.assertEqual(report["config_version"], CONFIG_VERSION)
        self.assertEqual(report["algorithm_version"], ALGORITHM_VERSION)
        self.assertEqual(report["config"], dict(self.inspector.config))
        for label in ("before", "after"):
            image = report["images"][label]
            validation = image["validation"]
            self.assertTrue(validation["valid"])
            self.assertTrue(all(validation["checks"].values()))
            self.assertEqual(validation["mode"], "RGB")
            self.assertEqual(validation["size"], [256, 256])
            self.assertEqual(validation["dtype"], "uint8")
            self.assertEqual(validation["observed_value_range"], [30, 225])
            self.assertNotIn("confidence", image)
            self.assertNotIn("is_blurry", image)
        self.assertIn("All input pixels are valid", report["limits"]["validity_assumption"])

    def test_blur_lowers_both_gradient_measurements(self):
        blurred = cv2.GaussianBlur(self.texture, (9, 9), 2.0)
        self._save(self.after, blurred)
        report = self.inspect()
        for name in ("laplacian_variance", "tenengrad_energy"):
            self.assertLess(report["images"]["after"]["sharpness"][name],
                            report["images"]["before"]["sharpness"][name] * 0.1)
        comparison = report["pair"]["sharpness_comparison"]
        self.assertEqual(comparison["direction"], "after_lower")
        self.assertEqual(comparison["assessment"], "review")
        self.assertIn("not a blur diagnosis", comparison["interpretation"])

    def test_noise_can_increase_gradients_without_a_quality_ranking(self):
        clean = cv2.GaussianBlur(self.texture, (9, 9), 2.0)
        noise = np.random.default_rng(31).normal(0, 20, clean.shape)
        noisy = np.clip(clean.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        self._save(self.before, clean)
        self._save(self.after, noisy)
        report = self.inspect()
        for name in ("laplacian_variance", "tenengrad_energy"):
            self.assertGreater(report["images"]["after"]["sharpness"][name],
                               report["images"]["before"]["sharpness"][name])
        comparison = report["pair"]["sharpness_comparison"]
        self.assertEqual(comparison["direction"], "before_lower")
        self.assertIn("not a blur diagnosis or quality ranking", comparison["interpretation"])
        self.assertIn("noise", comparison["interpretation"])

    def test_constant_black_white_and_gray_are_unknown_valid_samples(self):
        for level in (0, 128, 255):
            with self.subTest(level=level):
                pixels = np.full((256, 256, 3), level, dtype=np.uint8)
                self._save(self.before, pixels)
                self._save(self.after, pixels)
                report = self.inspect()
                image = report["images"]["before"]
                self.assertTrue(image["validation"]["valid"])
                self.assertEqual(image["brightness"]["mean"], level)
                self.assertEqual(image["brightness"]["variance"], 0.0)
                self.assertEqual(image["sharpness"]["laplacian_variance"], 0.0)
                self.assertEqual(image["sharpness"]["tenengrad_energy"], 0.0)
                self.assertEqual(image["assessment"], "unknown")
                self.assertEqual(image["texture_assessment"], "unknown")
                self.assertEqual(image["clipping"]["black_pixel_fraction"], float(level == 0))
                self.assertEqual(image["clipping"]["white_pixel_fraction"], float(level == 255))
                comparison = report["pair"]["sharpness_comparison"]
                self.assertEqual(comparison["direction"], "unknown")
                self.assertIsNone(comparison["laplacian_variance_after_over_before"])
                self.assertIn("NoData is not inferred", " ".join(report["limitations"]))

    def test_low_texture_image_remains_unknown(self):
        pixels = np.full((256, 256, 3), 128, dtype=np.uint8)
        pixels[:, 128:] = 129
        self._save(self.after, pixels)
        report = self.inspect()
        self.assertEqual(report["images"]["after"]["assessment"], "unknown")
        self.assertEqual(report["pair"]["sharpness_comparison"]["direction"], "unknown")

    def test_brightness_percentiles_and_channel_clipping_are_explicit(self):
        pixels = np.full((256, 256, 3), 64, dtype=np.uint8)
        pixels[:, 128:] = 192
        self._save(self.before, pixels)
        image = self.inspect()["images"]["before"]
        self.assertEqual(image["brightness"]["mean"], 128.0)
        self.assertEqual([image["brightness"][key] for key in ("p05", "p50", "p95")],
                         [64.0, 128.0, 192.0])
        pixels[:] = [0, 255, 128]
        self._save(self.before, pixels)
        image = self.inspect()["images"]["before"]
        self.assertAlmostEqual(image["clipping"]["channel_zero_fraction"], 1 / 3)
        self.assertAlmostEqual(image["clipping"]["channel_255_fraction"], 1 / 3)
        self.assertEqual(image["clipping"]["black_pixel_fraction"], 0.0)
        self.assertEqual(image["clipping"]["white_pixel_fraction"], 0.0)

    def test_64_pixel_tile_boxes_cover_image_once_with_exclusive_edges(self):
        tiles = self.inspect()["images"]["before"]["tiles"]
        self.assertEqual(len(tiles), 16)
        self.assertEqual(tiles[0]["bbox"], [0, 0, 64, 64])
        self.assertEqual(tiles[-1]["bbox"], [192, 192, 256, 256])
        coverage = np.zeros((256, 256), dtype=np.uint8)
        for index, tile in enumerate(tiles):
            self.assertEqual(tile["tile_index"], index)
            x0, y0, x1, y1 = tile["bbox"]
            self.assertEqual([x1 - x0, y1 - y0], [64, 64])
            coverage[y0:y1, x0:x1] += 1
            expected = self.texture[y0:y1, x0:x1]
            gray = cv2.cvtColor(expected, cv2.COLOR_RGB2GRAY)
            self.assertAlmostEqual(tile["brightness"]["mean"], float(gray.mean()))
            self.assertIn("tenengrad_energy", tile["sharpness"])
        self.assertTrue(np.all(coverage == 1))

    def test_jpeg_and_png_are_accepted_without_resize(self):
        jpeg = self.root / "B.jpg"
        self._save(jpeg, self.texture)
        report = self.inspector.inspect(str(self.before), str(jpeg))
        self.assertEqual(report["images"]["before"]["validation"]["format"], "PNG")
        self.assertEqual(report["images"]["after"]["validation"]["format"], "JPEG")

    def test_bad_path_types_missing_and_corrupt_images_raise_value_error(self):
        for value in ("", " ", None, True, 17, self.before, str(self.root / "missing.png"), str(self.root)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.inspector.inspect(value, str(self.after))
        corrupt = self.root / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            self.inspector.inspect(str(self.before), str(corrupt))

    def test_grayscale_rgba_palette_and_16bit_grayscale_are_rejected(self):
        for mode in ("L", "RGBA", "P", "I;16"):
            with self.subTest(mode=mode):
                path = self.root / (mode.replace(";", "_") + ".png")
                Image.new(mode, (256, 256)).save(path)
                with self.assertRaises(ValueError):
                    self.inspector.inspect(str(self.before), str(path))

    def test_16bit_rgb_png_is_rejected_before_pillow_downcasts(self):
        path = self.root / "rgb16.png"
        rgb16 = np.full((256, 256, 3), 32000, dtype=np.uint16)
        encoded, payload = cv2.imencode(".png", rgb16)
        self.assertTrue(encoded)
        path.write_bytes(payload.tobytes())
        with Image.open(path) as image:
            self.assertEqual(image.mode, "RGB")
        with self.assertRaisesRegex(ValueError, "8-bit RGB PNG"):
            self.inspector.inspect(str(path), str(self.after))

    def test_small_large_non_square_and_different_sizes_are_rejected(self):
        for size in ((64, 64), (255, 256), (256, 255), (512, 512)):
            with self.subTest(size=size):
                Image.new("RGB", size).save(self.after)
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_other_encoded_formats_are_rejected_even_with_png_extension(self):
        Image.new("RGB", (256, 256)).save(self.after, format="BMP")
        with self.assertRaises(ValueError):
            self.inspect()

    def test_explicit_nodata_png_text_metadata_is_rejected_case_insensitively(self):
        for key in ("NoData", "gdal_nodata", "GDAL_NoData", "no_data"):
            with self.subTest(key=key):
                metadata = PngInfo()
                metadata.add_text(key, "0")
                Image.fromarray(self.texture).save(self.after, pnginfo=metadata)
                with self.assertRaisesRegex(ValueError, "explicit NoData metadata"):
                    self.inspect()
                with self.assertRaisesRegex(ValueError, "explicit NoData metadata"):
                    self.inspector.inspect(str(self.after), str(self.before))

    def test_explicit_nodata_png_text_after_image_data_is_rejected(self):
        metadata = PngInfo()
        metadata.add_text("NoData", "0")
        Image.fromarray(self.texture).save(self.after, pnginfo=metadata)
        payload = self.after.read_bytes()
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
        chunks = []
        offset = 8
        while offset < len(payload):
            self.assertLessEqual(offset + 12, len(payload))
            length = int.from_bytes(payload[offset:offset + 4], "big")
            end = offset + length + 12
            self.assertLessEqual(end, len(payload))
            # Retain each entire chunk, including its original CRC.
            chunks.append(payload[offset:end])
            offset = end
        self.assertEqual(offset, len(payload))
        self.assertEqual(chunks[-1][4:8], b"IEND")
        text_chunks = [chunk for chunk in chunks if chunk[4:8] == b"tEXt"]
        self.assertEqual(len(text_chunks), 1)
        self.assertEqual(text_chunks[0][8:-4], b"NoData\x000")
        image_chunks = [chunk for chunk in chunks[:-1] if chunk[4:8] != b"tEXt"]
        self.assertIn(b"IDAT", [chunk[4:8] for chunk in image_chunks])
        self.after.write_bytes(payload[:8] + b"".join(image_chunks + text_chunks + chunks[-1:]))
        with Image.open(self.after) as image:
            self.assertNotIn("NoData", image.info)
            image.load()
            self.assertEqual(image.info["NoData"], "0")
        with self.assertRaisesRegex(ValueError, "explicit NoData metadata"):
            self.inspect()

    def test_config_is_validated_copied_and_frozen(self):
        invalid = [[], "profile", {"unknown": 1}, {"tile_size": True},
                   {"tile_size": 64.0}, {"tile_size": 32},
                   {"min_variance": True}, {"min_variance": -1},
                   {"min_variance": 65026}, {"min_variance": "4"},
                   {"min_variance": float("nan")}, {"min_variance": float("inf")},
                   {"min_variance": 10 ** 999}, {"max_clipped_fraction": 1.01},
                   {"max_clipped_fraction": -0.1}, {"min_mean_brightness": -1},
                   {"max_mean_brightness": 256}, {"min_mean_brightness": 240},
                   {"relative_sharpness_ratio": 0}, {"relative_sharpness_ratio": 1},
                   {"min_laplacian_variance": -1}, {"min_tenengrad_energy": -1},
                   {"max_clipped_fraction": np.bool_(True)}]
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    ImageQualityInspector(config)
        original = {"min_variance": 8}
        inspector = ImageQualityInspector(original)
        original["min_variance"] = 99
        self.assertEqual(inspector.config["min_variance"], 8)
        with self.assertRaises(TypeError):
            inspector.config["min_variance"] = 99
        report = inspector.inspect(str(self.before), str(self.after))
        report["config"]["min_variance"] = 99
        self.assertEqual(inspector.config["min_variance"], 8)

    def test_zero_threshold_cannot_make_constant_image_measurable(self):
        self._save(self.after, np.full((256, 256, 3), 128, dtype=np.uint8))
        inspector = ImageQualityInspector({"min_variance": 0})
        report = inspector.inspect(str(self.before), str(self.after))
        self.assertEqual(report["images"]["after"]["texture_assessment"], "unknown")


if __name__ == "__main__":
    unittest.main()
