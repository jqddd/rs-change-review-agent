"""Offline contract tests: fake detector results are not pretrained validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from lagent.actions import ActionExecutor
from lagent.agents import ReAct
from lagent.schema import ActionStatusCode

from algorithms.ddpm_cd.inference import DDPMCD, PROFILE, fixed_noise, load_state, make_diffusion, make_head
from tools.ddpm_cd_tool import DDPMCDTool


class FakeDetector:
    device = torch.device("cpu")
    configuration = {"test_fixture": True}

    def detect(self, before, after):
        mask = np.zeros((256, 256), np.uint8)
        mask[1:11, 2:12] = 1
        return mask


class DDPMCDToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / "A.png", self.root / "B.png"
        for path in (self.a, self.b):
            Image.new("RGB", (256, 256), (255, 0, 127)).save(path)
        self.args = dict(image_before=str(self.a), image_after=str(self.b), output_dir=str(self.root / "out"))
        self.tool = DDPMCDTool(device="cpu")

    def tearDown(self):
        self.temp.cleanup()

    def test_json_contract_and_unique_outputs(self):
        executor = ActionExecutor(actions=[self.tool])
        with patch.object(self.tool, "_get_model", return_value=FakeDetector()):
            first = executor("ddpm_cd", json.dumps(self.args))
            second = executor("ddpm_cd", self.args)
        self.assertEqual(first.state, ActionStatusCode.SUCCESS)
        payload = json.loads(first.result[0]["content"])
        self.assertEqual(payload["diagnostics"]["changed_pixel_count"], 100)
        self.assertNotEqual(payload["run_id"], json.loads(second.result[0]["content"])["run_id"])
        label = Path(payload["artifacts"][0]["path"])
        self.assertTrue(label.is_absolute())
        with Image.open(label) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(set(np.asarray(image).flatten()), {0, 1})
        self.assertTrue((label.parent / "result.json").is_file())

    def test_bad_images_rejected_before_model_load(self):
        for mode, size in [("L", (256, 256)), ("RGB", (512, 512))]:
            Image.new(mode, size).save(self.a)
            self.assertEqual(self.tool(self.args).state, ActionStatusCode.ARGS_ERROR)
        self.a.unlink()
        self.assertEqual(self.tool(self.args).state, ActionStatusCode.ARGS_ERROR)
        self.assertIsNone(self.tool._model)

    def test_missing_checkpoint_and_failed_write_not_success(self):
        with patch.object(self.tool, "_get_model", side_effect=FileNotFoundError("missing weights")):
            self.assertEqual(self.tool(self.args).state, ActionStatusCode.API_ERROR)
        with patch.object(self.tool, "_get_model", return_value=FakeDetector()), patch.object(Image.Image, "save", side_effect=OSError("disk full")):
            self.assertEqual(self.tool(self.args).state, ActionStatusCode.API_ERROR)
        with self.assertRaises(FileNotFoundError):
            DDPMCD(self.root / "none.pth", self.root / "no_head.pth", "cpu")

    def test_preprocessing_and_checkpoint_validation(self):
        tensor = DDPMCD._preprocess(self.a)
        self.assertEqual(tuple(tensor.shape), (1, 3, 256, 256))
        torch.testing.assert_close(tensor[0, :, 0, 0], torch.tensor([1., -1., 127/255*2-1]))
        layer = torch.nn.Linear(2, 1)
        checkpoint = self.root / "tiny.pth"
        torch.save({"module." + k: v for k, v in layer.state_dict().items()}, checkpoint)
        with torch.device("meta"):
            rebuilt = torch.nn.Linear(2, 1)
        load_state(checkpoint, rebuilt)
        torch.testing.assert_close(layer.weight, rebuilt.weight)
        # Windows cannot overwrite an mmap-backed file while tensors reference it.
        legacy_checkpoint = self.root / "tiny_legacy.pth"
        torch.save(layer.state_dict(), legacy_checkpoint, _use_new_zipfile_serialization=False)
        load_state(legacy_checkpoint, rebuilt)
        torch.testing.assert_close(layer.weight, rebuilt.weight)
        bad_checkpoint = self.root / "invalid.pth"
        torch.save({"unknown": torch.ones(1)}, bad_checkpoint)
        with self.assertRaises(RuntimeError):
            load_state(bad_checkpoint, rebuilt)

    def test_two_checkpoint_lazy_load_and_partial_download_guard(self):
        fake = FakeDetector()
        fake.diffusion_weight = self.root / 'diffusion.pth'
        fake.change_weight = self.root / 'head.pth'
        fake.diffusion_weight.write_bytes(b'diffusion test fixture')
        fake.change_weight.write_bytes(b'head test fixture')
        with patch('tools.ddpm_cd_tool.DDPMCD', return_value=fake) as factory:
            self.assertIs(self.tool._get_model(), fake)
            self.assertIs(self.tool._get_model(), fake)
            factory.assert_called_once()
            self.assertEqual(set(self.tool._weights), {'diffusion', 'change_head'})
        partial = self.root / PROFILE['diffusion_file']
        partial.write_bytes(b'partial')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            DDPMCD(partial, fake.change_weight, device='cpu')

    def test_rng_repeatability_restoration_and_exception(self):
        cpu_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()
        values = []
        for _ in range(2):
            with fixed_noise(17, torch.device("cpu")):
                values.append((np.random.uniform(), torch.randn(3)))
        self.assertEqual(values[0][0], values[1][0])
        torch.testing.assert_close(values[0][1], values[1][1], rtol=0, atol=0)
        self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
        np.testing.assert_equal(numpy_state, np.random.get_state())
        with self.assertRaisesRegex(RuntimeError, "test"):
            with fixed_noise(22, torch.device("cpu")):
                raise RuntimeError("test")
        self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
        np.testing.assert_equal(numpy_state, np.random.get_state())

    def test_react_feedback_uses_registered_tool(self):
        llm = Mock()
        llm.chat.side_effect = [
            "Thought: use tool\nAction: ddpm_cd\nAction Input: " + json.dumps(self.args),
            "Thought: done\nFinal Answer: completed",
        ]
        with patch.object(self.tool, "_get_model", return_value=FakeDetector()):
            result = ReAct(llm, ActionExecutor(actions=[self.tool])).chat("Detect building change")
        self.assertEqual(result.actions[0].type, "ddpm_cd")
        self.assertEqual(result.actions[0].state, ActionStatusCode.SUCCESS)
        self.assertIn("changed_pixel_count", str(llm.chat.call_args_list[1].args[0]))

    def test_official_profile_meta_shapes_and_namespaces(self):
        # No GPU allocation and no random full-size weights: structure only.
        with torch.device("meta"):
            diffusion, head = make_diffusion(PROFILE), make_head(PROFILE)
        self.assertEqual(sum(p.numel() for p in diffusion.parameters()), 391048067)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 46405126)
        self.assertEqual(PROFILE["timesteps"], [50, 100, 400])
        self.assertNotIn("inference", sys.modules)
        self.assertNotIn("model.sr3_modules.unet", sys.modules)


if __name__ == "__main__":
    unittest.main()
