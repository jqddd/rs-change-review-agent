"""Offline contract tests and explicitly untrained architecture smoke test."""
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from lagent.actions import ActionExecutor
from lagent.agents import ReAct
from lagent.schema import ActionStatusCode

from algorithms.snunet.inference import SNUNet, validate_image
from algorithms.snunet.models.Models import SNUNet_ECAM
from algorithms.snunet.models import Models as upstream_model
from algorithms.snunet.convert_checkpoint import convert, _UpstreamUnpickler
from tools.snunet_tool import SNUNetTool


class FakeDetector:
    device = torch.device('cpu')
    configuration = {}

    def detect(self, before, after):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:20, 30:40] = 1
        return mask


class SNUNetContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.before, self.after = self.root / 'A.png', self.root / 'B.png'
        for path in (self.before, self.after):
            Image.new('RGB', (256, 256), (255, 127, 0)).save(path)
        self.tool = SNUNetTool(device='cpu', profile='cdd_c32')
        self.args = dict(image_before=str(self.before), image_after=str(self.after), output_dir=str(self.root / 'out'))

    def tearDown(self):
        self.tmp.cleanup()

    def test_executor_contract_and_unique_outputs(self):
        executor = ActionExecutor(actions=[self.tool])
        with patch.object(self.tool, '_get_model', return_value=FakeDetector()):
            first = executor('snunet', json.dumps(self.args))
            second = executor('snunet', self.args)
        self.assertEqual(first.state, ActionStatusCode.SUCCESS)
        payload = json.loads(first.result[0]['content'])
        payload2 = json.loads(second.result[0]['content'])
        self.assertEqual(payload['schema_version'], '0.1')
        self.assertEqual(payload['diagnostics']['changed_pixel_count'], 100)
        self.assertNotEqual(payload['run_id'], payload2['run_id'])
        label = Path(payload['artifacts'][0]['path'])
        self.assertTrue(label.is_absolute())
        with Image.open(label) as image:
            self.assertEqual(image.mode, 'L')
            self.assertEqual(set(np.asarray(image).flatten()), {0, 1})
        self.assertTrue((label.parent / 'result.json').is_file())
        self.assertEqual(payload['artifacts'][0]['reference_image'], str(self.before))
        # Inputs remain untouched.
        with Image.open(self.before) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 127, 0))

    def test_missing_wrong_size_and_wrong_channel(self):
        for mode, size in [('RGB', (64, 64)), ('L', (256, 256))]:
            Image.new(mode, size).save(self.before)
            result = self.tool(self.args)
            self.assertEqual(result.state, ActionStatusCode.ARGS_ERROR)
        self.before.unlink()
        self.assertEqual(self.tool(self.args).state, ActionStatusCode.ARGS_ERROR)
        self.assertIsNone(self.tool._model)

    def test_inference_and_output_failures_are_not_success(self):
        with patch.object(self.tool, '_get_model', side_effect=RuntimeError('missing weights')):
            self.assertEqual(self.tool(self.args).state, ActionStatusCode.API_ERROR)
        with patch.object(self.tool, '_get_model', return_value=FakeDetector()), patch.object(Image.Image, 'save', side_effect=OSError('disk full')):
            self.assertEqual(self.tool(self.args).state, ActionStatusCode.API_ERROR)

    def test_registration_and_react_feedback_without_real_llm(self):
        # Scripted responses test orchestration only, not DeepSeek's decisions.
        from unittest.mock import Mock
        llm = Mock()
        llm.chat.side_effect = [
            'Thought: use tool\nAction: snunet\nAction Input: ' + json.dumps(self.args),
            'Thought: done\nFinal Answer: completed',
        ]
        with patch.object(self.tool, '_get_model', return_value=FakeDetector()):
            result = ReAct(llm, ActionExecutor(actions=[self.tool])).chat('Detect change')
        self.assertEqual(result.actions[0].type, 'snunet')
        self.assertEqual(result.actions[0].state, ActionStatusCode.SUCCESS)
        second_prompt = llm.chat.call_args_list[1].args[0]
        self.assertIn('changed_pixel_count', str(second_prompt))

    def test_preprocessing_matches_upstream_not_imagenet(self):
        tensor = SNUNet._preprocess(validate_image(str(self.before)), profile='cdd_c32')
        self.assertEqual(tuple(tensor.shape), (1, 3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [255.0, 127.0, 0.0])

    def test_random_weights_structure_smoke_and_legacy_conversion(self):
        # RANDOM parameters: proves compatibility, not model quality.
        weight, legacy = self.root / 'random.pth', self.root / 'random.pt'
        model = SNUNet_ECAM(in_ch=3, out_ch=2)
        for name in ('ca', 'ca1'):
            getattr(model, name).max_pool = torch.nn.AdaptiveAvgPool2d(1)
        # Official CDD checkpoint also stores these modules, unused by forward.
        for index in (2, 3, 4):
            model.add_module(f'ca{index}', upstream_model.ChannelAttention(32, ratio=4))
        # Emulate upstream module names AND the pre-PyTorch-1.6 pickle format.
        # Temporary aliases exist only while producing this test fixture.
        legacy_package = types.ModuleType('models')
        legacy_package.__path__ = []
        legacy_module = types.ModuleType('models.Models')
        legacy_package.Models = legacy_module
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'models': legacy_package, 'models.Models': legacy_module}))
            for name in ('SNUNet_ECAM', 'conv_block_nested', 'up', 'ChannelAttention'):
                cls = getattr(upstream_model, name)
                setattr(legacy_module, name, cls)
                stack.enter_context(patch.object(cls, '__module__', 'models.Models'))
            torch.save(model, legacy, _use_new_zipfile_serialization=False)
        convert(legacy, weight)
        converted = torch.load(weight, map_location='cpu', weights_only=True)
        self.assertEqual(set(converted['unused_legacy_state']),
                         {f'ca{i}.fc{j}.weight' for i in (2, 3, 4) for j in (1, 2)})
        for name, tensor in converted['unused_legacy_state'].items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[name]))
        self.assertFalse(any(key.startswith(('ca2.', 'ca3.', 'ca4.')) for key in converted['state_dict']))
        detector = SNUNet(weight, device='cpu', profile='cdd_c32')
        self.assertIsInstance(detector.model.ca.max_pool, torch.nn.AdaptiveAvgPool2d)
        self.assertIsInstance(detector.model.ca1.max_pool, torch.nn.AdaptiveAvgPool2d)
        # state_dict equality alone cannot catch parameter-free architecture drift.
        original = model.to(detector.device).eval()
        generator = torch.Generator(device='cpu').manual_seed(27)
        a = (torch.rand((1, 3, 64, 64), generator=generator) * 255).to(detector.device)
        b = (torch.rand((1, 3, 64, 64), generator=generator) * 255).to(detector.device)
        with torch.inference_mode():
            torch.testing.assert_close(original(a, b)[-1], detector.model(a, b)[-1], rtol=0, atol=0)
        mask = detector.detect(str(self.before), str(self.after))
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.isin(mask, [0, 1]).all())
        # Check legacy alias resolution without introducing generic modules.
        import io
        self.assertIs(_UpstreamUnpickler(io.BytesIO()).find_class('models.Models', 'SNUNet_ECAM'), SNUNet_ECAM)
        self.assertNotIn('inference', sys.modules)
        self.assertNotIn('models.Models', sys.modules)
        with patch('tools.snunet_tool.SNUNet', return_value=detector) as factory:
            tool = SNUNetTool(str(weight), device='cpu', profile='cdd_c32')
            self.assertIs(tool._get_model(), tool._get_model())
            factory.assert_called_once()

    def test_conversion_still_rejects_unknown_parameters(self):
        legacy = self.root / 'unexpected.pt'
        model = SNUNet_ECAM(in_ch=3, out_ch=2)
        model.register_parameter('unrecognized_weight', torch.nn.Parameter(torch.ones(1)))
        torch.save(model, legacy)
        with self.assertRaisesRegex(ValueError, 'Unexpected checkpoint keys'):
            convert(legacy, self.root / 'must_not_exist.pth')
        self.assertFalse((self.root / 'must_not_exist.pth').exists())


if __name__ == '__main__':
    unittest.main()
