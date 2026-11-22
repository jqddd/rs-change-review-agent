"""Offline LEVIR C16 deployment tests; random weights do not measure accuracy."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from lagent.actions import ActionExecutor
from lagent.schema import ActionStatusCode

from algorithms.snunet.convert_opencd_checkpoint import _extract, convert
from algorithms.snunet.inference import SNUNet
from algorithms.snunet.models.opencd_c16 import SNUNetC16, UPSTREAM_COMMIT
from tools.snunet_tool import SNUNetTool


class SNUNetC16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        # A complete local random fixture, with the Open-CD checkpoint names.
        # It is never presented as a trained LEVIR-CD model.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(61)
            cls.state = SNUNetC16().state_dict()

    @classmethod
    def tearDownClass(cls):
        del cls.state
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.before, self.after = self.root / 'A.png', self.root / 'B.png'
        Image.new('RGB', (256, 256), (255, 127, 0)).save(self.before)
        Image.new('RGB', (256, 256), (12, 64, 240)).save(self.after)
        self.weight = self.root / 'random_c16.pth'
        self.args = dict(image_before=str(self.before), image_after=str(self.after),
                         output_dir=str(self.root / 'out'))

    def tearDown(self):
        self.tmp.cleanup()

    def write_checkpoint(self):
        torch.save({'profile': 'levir_c16', 'state_dict': self.state,
                    'source_sha256': 'local-random-fixture'}, self.weight)

    def load_patched_checkpoint(self, checkpoint, **kwargs):
        # The loading boundary still requires an explicit existing file.
        self.weight.touch()
        with patch('algorithms.snunet.inference.torch.load', return_value=checkpoint):
            return SNUNet(self.weight, device='cpu', **kwargs)

    def test_rgb_mean_std_on_raw_pixels_without_resizing(self):
        actual = SNUNet._preprocess(self.before)
        self.assertEqual(tuple(actual.shape), (1, 3, 256, 256))
        self.assertEqual(actual.dtype, torch.float32)
        # Independent reference values for RGB=(255,127,0), not /255 or BGR.
        expected = torch.tensor([2.2489083, 0.1876751, -1.8044444])
        torch.testing.assert_close(actual, expected[None, :, None, None].expand_as(actual),
                                   rtol=0, atol=1e-6)
        legacy = SNUNet._preprocess(self.before, profile='cdd_c32')
        self.assertEqual(legacy[0, :, 0, 0].tolist(), [255.0, 127.0, 0.0])

    def test_real_small_forward_and_opencd_architecture_configuration(self):
        model = SNUNetC16().eval()
        generator = torch.Generator().manual_seed(19)
        before = torch.randn((1, 3, 32, 32), generator=generator)
        after = torch.randn((1, 3, 32, 32), generator=generator)
        with torch.inference_mode():
            output = model(before, after)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (1, 2, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(self.state['backbone.conv0_0.conv1.weight'].shape), (16, 3, 3, 3))
        self.assertEqual(tuple(self.state['decode_head.conv_seg.weight'].shape), (2, 64, 1, 1))
        self.assertFalse(any(key.startswith('backbone.conv_final.') for key in self.state))
        # Pinned checkpoint inventory, including BN running statistics/buffers.
        self.assertEqual(len(self.state), 236)
        self.assertEqual(sum(tensor.numel() for tensor in self.state.values()), 3_015_856)
        self.assertIsInstance(model.backbone.ca.max_pool, torch.nn.AdaptiveMaxPool2d)
        self.assertIsInstance(model.backbone.ca1.max_pool, torch.nn.AdaptiveMaxPool2d)
        batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        self.assertTrue(batch_norms)
        for batch_norm in batch_norms:
            self.assertEqual(batch_norm.eps, 1e-5)
            self.assertEqual(batch_norm.momentum, 0.1)

    def test_converted_tensor_checkpoint_loads_on_cpu_and_uses_eval(self):
        self.write_checkpoint()
        with patch('algorithms.snunet.inference.torch.load', wraps=torch.load) as loader:
            detector = SNUNet(self.weight, device='cpu')
        loader.assert_called_once_with(self.weight.resolve(), map_location='cpu', weights_only=True)
        self.assertEqual(detector.profile, 'levir_c16')
        self.assertIsInstance(detector.model, SNUNetC16)
        self.assertFalse(detector.model.training)
        self.assertTrue(all(parameter.device.type == 'cpu' for parameter in detector.model.parameters()))
        self.assertEqual(detector.configuration['trained_on'], 'LEVIR-CD')
        self.assertEqual(detector.configuration['base_channel'], 16)
        self.assertEqual(detector.configuration['source_sha256'], 'local-random-fixture')
        torch.testing.assert_close(detector.model.state_dict()['decode_head.conv_seg.weight'],
                                   self.state['decode_head.conv_seg.weight'], rtol=0, atol=0)

    def test_loading_is_strict_for_keys_and_shapes(self):
        key = 'decode_head.conv_seg.weight'
        missing = dict(self.state)
        del missing[key]
        extra = dict(self.state, unknown_weight=torch.zeros(1))
        wrong_shape = dict(self.state)
        wrong_shape[key] = torch.zeros((3, 64, 1, 1))
        for state, message in ((missing, 'Missing key'), (extra, 'Unexpected key'),
                               (wrong_shape, 'size mismatch')):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                self.load_patched_checkpoint({'profile': 'levir_c16', 'state_dict': state})

    def test_profile_metadata_is_required_and_cannot_cross_load(self):
        for checkpoint in (self.state, {'state_dict': self.state},
                           {'profile': 'cdd_c32', 'state_dict': self.state}):
            with self.subTest(profile=getattr(checkpoint, 'get', lambda *_: None)('profile')):
                with self.assertRaisesRegex(ValueError, 'profile'):
                    self.load_patched_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, 'profile'):
            self.load_patched_checkpoint({'profile': 'levir_c16', 'state_dict': self.state},
                                         profile='cdd_c32')
        for invalid_state in ({}, {'weight': 'not a tensor'}):
            with self.subTest(state=invalid_state), self.assertRaisesRegex(ValueError, 'tensor state_dict'):
                self.load_patched_checkpoint({'profile': 'levir_c16', 'state_dict': invalid_state})

    def test_detect_uses_c16_logits_argmax_and_profile_normalization(self):
        detector = self.load_patched_checkpoint({'profile': 'levir_c16', 'state_dict': self.state})
        logits = torch.zeros((1, 2, 256, 256))
        logits[:, 1, 10:20, 30:40] = 2
        with patch.object(detector.model, 'forward', return_value=logits) as forward:
            mask = detector.detect(str(self.before), str(self.after))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(int(mask.sum()), 100)
        self.assertTrue((mask[10:20, 30:40] == 1).all())
        for actual, path in zip(forward.call_args.args, (self.before, self.after)):
            torch.testing.assert_close(actual, SNUNet._preprocess(path), rtol=0, atol=0)
        for invalid in (torch.zeros((1, 2, 32, 32)), torch.full_like(logits, float('nan'))):
            with self.subTest(shape=invalid.shape), patch.object(detector.model, 'forward', return_value=invalid):
                with self.assertRaisesRegex(ValueError, 'logits'):
                    detector.detect(str(self.before), str(self.after))

    def test_default_tool_capability_labels_and_provenance(self):
        self.write_checkpoint()
        detector = SNUNet(self.weight, device='cpu')
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[3:5, 6:8] = 1
        tool = SNUNetTool(str(self.weight), device='cpu')
        executor = ActionExecutor(actions=[tool])
        with patch('tools.snunet_tool.SNUNet', return_value=detector) as factory, \
                patch.object(detector, 'detect', return_value=mask):
            first = executor('snunet', json.dumps(self.args))
            second = executor('snunet', self.args)
        factory.assert_called_once_with(self.weight.resolve(), 'cpu', profile='levir_c16')
        self.assertEqual(first.state, ActionStatusCode.SUCCESS)
        self.assertEqual(second.state, ActionStatusCode.SUCCESS)
        result = json.loads(first.result[0]['content'])
        cached_result = json.loads(second.result[0]['content'])
        self.assertNotEqual(result['run_id'], cached_result['run_id'])
        label = result['artifacts'][0]
        self.assertEqual(label['labels'], {'0': 'unchanged', '1': 'building_change'})
        with Image.open(label['path']) as image:
            self.assertEqual(image.mode, 'L')
            np.testing.assert_array_equal(np.asarray(image), mask)
        provenance = result['provenance']
        self.assertEqual(provenance['profile'], 'levir_c16')
        self.assertEqual(provenance['capability']['target_object'], 'building')
        self.assertEqual(provenance['capability']['trained_on'], 'LEVIR-CD')
        self.assertEqual(provenance['weight'], str(self.weight.resolve()))
        self.assertEqual(provenance['weight_sha256'], hashlib.sha256(self.weight.read_bytes()).hexdigest())
        self.assertEqual(provenance['upstream_commit'], UPSTREAM_COMMIT)
        self.assertEqual(provenance['model_configuration']['source_sha256'], 'local-random-fixture')
        self.assertIn('subtract RGB mean', provenance['preprocessing'])
        self.assertTrue(provenance['includes_model_load'])
        self.assertFalse(cached_result['provenance']['includes_model_load'])
        self.assertTrue(any('1024' in warning for warning in result['warnings']))
        self.assertIn('LEVIR-CD', tool.description['description'])
        self.assertIn('building', tool.description['description'])
        self.assertEqual(json.loads((Path(label['path']).parent / 'result.json').read_text(encoding='utf-8')), result)

    def test_profile_selection_and_training_domain_are_bound(self):
        default = SNUNetTool(device='cpu')
        legacy = SNUNetTool(device='cpu', profile='cdd_c32', trained_on='CDD')
        self.assertEqual(default.profile, 'levir_c16')
        self.assertEqual(default.weight_path.name, 'snunet_levir_c16.pth')
        self.assertEqual(default.capability['trained_on'], 'LEVIR-CD')
        self.assertEqual(default.capability['target_object'], 'building')
        self.assertEqual(legacy.weight_path.name, 'snunet_cdd.pth')
        self.assertEqual(legacy.capability['target_object'], 'general_change')
        self.assertEqual(legacy.settings['labels']['1'], 'change')
        for kwargs in ({'trained_on': 'CDD'}, {'profile': 'cdd_c32', 'trained_on': 'LEVIR-CD'},
                       {'profile': 'unknown'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SNUNetTool(device='cpu', **kwargs)
        with self.assertRaisesRegex(ValueError, 'profile'):
            SNUNet(self.weight, device='cpu', profile='unknown')

    def test_missing_weights_and_invalid_tool_parameters_are_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, 'WEIGHTS.md'):
            SNUNet(self.weight, device='cpu')
        tool = SNUNetTool(str(self.weight), device='cpu')
        with patch.object(tool, '_get_model') as load:
            for args in ({}, {'image_before': str(self.before)},
                         dict(self.args, profile='cdd_c32'),
                         dict(self.args, image_after=str(self.root / 'missing.png')),
                         dict(self.args, output_dir=''), dict(self.args, output_dir=None),
                         dict(self.args, output_dir=str(self.before))):
                with self.subTest(args=args):
                    self.assertEqual(tool(args).state, ActionStatusCode.ARGS_ERROR)
            load.assert_not_called()
        self.assertEqual(tool(self.args).state, ActionStatusCode.API_ERROR)
        self.assertIsNone(tool._model)
        self.assertFalse((self.root / 'out').exists())


class _MustRemainInert:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        # A normal unpickler would create this file. The converter must only
        # record GLOBAL/REDUCE as data and never execute this expression.
        return (eval, (f"__import__('pathlib').Path({str(self.marker)!r}).write_text('executed')",))


class OpenCDConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_inference_rejects_pickled_objects_without_unsafe_fallback(self):
        source, marker = self.root / 'unsafe.pth', self.root / 'must-not-exist.txt'
        torch.save({'profile': 'levir_c16', 'state_dict': {'weight': torch.ones(1)},
                    'unneeded': _MustRemainInert(marker)}, source, pickle_protocol=2)
        with self.assertRaisesRegex(ValueError, 'tensor-only'):
            SNUNet(source, device='cpu')
        self.assertFalse(marker.exists())

    def test_extracts_only_tensors_without_executing_checkpoint_objects(self):
        source, marker = self.root / 'synthetic.pth', self.root / 'must-not-exist.txt'
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        state = {'matrix': base[:, ::2], 'shared': base, 'scalar': torch.tensor(4, dtype=torch.int64),
                 'empty': torch.empty((0, 2))}
        torch.save({'state_dict': state, 'meta': {'cfg': 'synthetic configuration', 'iter': 2},
                    'unneeded': _MustRemainInert(marker)}, source, pickle_protocol=2)
        extracted, config, meta = _extract(source)
        self.assertFalse(marker.exists())
        self.assertEqual(config, 'synthetic configuration')
        self.assertEqual(meta, {'iter': 2})
        self.assertEqual(set(extracted), set(state))
        for name, tensor in extracted.items():
            self.assertEqual(tensor.device.type, 'cpu')
            torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)

    def test_converter_requires_pinned_source_and_refuses_overwrite(self):
        source, output = self.root / 'unknown.pth', self.root / 'output.pth'
        source.write_bytes(b'not the pinned checkpoint')
        with self.assertRaisesRegex(ValueError, 'pinned'):
            convert(source, output)
        self.assertFalse(output.exists())
        with patch('algorithms.snunet.convert_opencd_checkpoint.SOURCE_SIZE', source.stat().st_size), \
                patch('algorithms.snunet.convert_opencd_checkpoint._extract') as extract:
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                convert(source, output)
            extract.assert_not_called()
        self.assertFalse(output.exists())
        output.write_bytes(b'preserve existing output')
        with self.assertRaises(FileExistsError):
            convert(source, output)
        self.assertEqual(output.read_bytes(), b'preserve existing output')

    def test_extractor_rejects_corrupt_tensor_storage(self):
        source, corrupt = self.root / 'source.pth', self.root / 'corrupt.pth'
        torch.save({'state_dict': {'weight': torch.ones(2)}, 'meta': {'cfg': 'synthetic'}}, source,
                   pickle_protocol=2)
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(corrupt, 'w') as rewritten:
            for info in original.infolist():
                data = original.read(info.filename)
                if info.filename.endswith('/data/0'):
                    data = data[:-1]
                rewritten.writestr(info.filename, data, compress_type=zipfile.ZIP_STORED)
        with self.assertRaisesRegex(ValueError, 'Storage size mismatch'):
            _extract(corrupt)


if __name__ == '__main__':
    unittest.main()
