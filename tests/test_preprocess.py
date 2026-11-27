"""CPU tests for module 4 preprocessing/repair primitives and their tools."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from lagent.schema import ActionStatusCode

from algorithms.preprocess import deblur, denoise, histogram_match
from tools.preprocess_tool import DeblurTool, DenoiseTool, HistogramMatchTool


def _texture(seed=17, brightness=0):
    rng = np.random.default_rng(seed)
    image = rng.integers(30, 226, size=(256, 256, 3), dtype=np.uint8)
    if brightness:
        image = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return image


def _save(path, array):
    Image.fromarray(array).save(path)


class PreprocessPrimitiveTests(unittest.TestCase):
    def test_histogram_match_shapes(self):
        out = histogram_match(_texture(1), _texture(2, brightness=60))
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_histogram_match_reduces_pair_diff(self):
        before, after = _texture(1), _texture(1, brightness=60)
        matched = histogram_match(before, after)
        raw_diff = float(np.abs(before.astype(np.int16) - after.astype(np.int16)).mean())
        matched_diff = float(np.abs(before.astype(np.int16) - matched.astype(np.int16)).mean())
        self.assertLess(matched_diff, raw_diff)

    def test_denoise_methods(self):
        image = _texture(3)
        for method in ("nlmeans", "bilateral", "median", "gaussian"):
            out = denoise(image, method=method)
            self.assertEqual(out.shape, (256, 256, 3), method)
            self.assertEqual(out.dtype, np.uint8, method)

    def test_default_nlmeans_matches_opencv_bgr_reference(self):
        import cv2
        rgb = _texture(19)[:32, :32].copy()
        rgb[:, :16, 0] //= 4
        rgb[:, 16:, 2] //= 4
        expected_bgr = cv2.fastNlMeansDenoisingColored(
            rgb[:, :, ::-1].copy(), None, 10, 10, 7, 21)
        np.testing.assert_array_equal(denoise(rgb), expected_bgr[:, :, ::-1])

    def test_denoise_rejects_bad_method(self):
        with self.assertRaises(ValueError):
            denoise(_texture(), method="bogus")

    def test_deblur(self):
        out = deblur(_texture(4))
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual(out.dtype, np.uint8)


class PreprocessToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test_preprocess_")
        self.root = Path(self.temp.name)
        self.before = self.root / "A.png"
        self.after = self.root / "B.png"
        _save(self.before, _texture(5))
        _save(self.after, _texture(5, brightness=60))
        self.out = self.root / "out"

    def tearDown(self):
        self.temp.cleanup()

    def test_histogram_match_tool_contract(self):
        result = HistogramMatchTool().run(str(self.before), str(self.after), str(self.out))
        self.assertIsInstance(result, dict)
        for key in ("schema_version", "run_id", "summary", "artifacts",
                    "diagnostics", "warnings", "elapsed_seconds", "provenance"):
            self.assertIn(key, result)
        kinds = {a["kind"] for a in result["artifacts"]}
        self.assertIn("repaired_image_after", kinds)
        self.assertIn("repaired_pair", kinds)
        pair = result["artifacts"][next(i for i, a in enumerate(result["artifacts"])
                                        if a["kind"] == "repaired_pair")]["path"]
        self.assertTrue(Path(pair).is_file())

    def test_denoise_tool(self):
        result = DenoiseTool().run(str(self.before), str(self.after), str(self.out), method="median")
        self.assertIsInstance(result, dict)
        self.assertIn("denoise", result["summary"])

    def test_denoise_distinguishes_black_pixels_from_declared_nodata(self):
        image = Image.fromarray(_texture(5))
        image.paste((0, 0, 0), (250, 0, 256, 256))
        image.save(self.after)
        tool = DenoiseTool()
        result = tool.run(str(self.before), str(self.after), str(self.out), method="median")
        self.assertIsInstance(result, dict)

        valid = Image.new("L", image.size, 1)
        valid.paste(0, (250, 0, 256, 256))
        valid.save(self.root / "valid_mask.png")
        info = PngInfo()
        info.add_text("NoData_mask", str(self.root / "valid_mask.png"))
        image.save(self.after, pnginfo=info)
        refused_dir = self.root / "refused"
        result = tool.run(str(self.before), str(self.after), str(refused_dir), method="median")
        self.assertEqual(getattr(result, "state", None), ActionStatusCode.ARGS_ERROR)
        self.assertIn("NoData", result.errmsg)
        self.assertFalse(refused_dir.exists())

    def test_deblur_tool(self):
        result = DeblurTool().run(str(self.before), str(self.after), str(self.out))
        self.assertIsInstance(result, dict)
        self.assertIn("deblur", result["summary"])

    def test_missing_input_is_args_error(self):
        from lagent.schema import ActionReturn
        result = HistogramMatchTool().run(str(self.root / "missing.png"), str(self.after), str(self.out))
        self.assertIsInstance(result, ActionReturn)


if __name__ == "__main__":
    unittest.main()
