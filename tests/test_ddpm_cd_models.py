"""Random small-model compatibility checks for vendored DDPM-CD modules.

These tests do not load pretrained weights or measure detection accuracy.
The optional meta-device structure check does not demonstrate actual 8GB GPU
memory compatibility; only a real checkpoint run can establish that.
"""
from contextlib import contextmanager
import unittest

import numpy as np
import torch

from algorithms.ddpm_cd.models.cd_modules.cd_head_v2 import cd_head_v2
from algorithms.ddpm_cd.models.sr3_modules.diffusion import GaussianDiffusion
from algorithms.ddpm_cd.models.sr3_modules.unet import UNet


@contextmanager
def isolated_seed(seed):
    """Seed the two RNGs used by upstream feats(), restoring caller state."""
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            np.random.seed(seed)
            # These tests allocate CPU tensors only; avoid touching CUDA RNGs.
            torch.random.default_generator.manual_seed(seed)
            yield
        finally:
            np.random.set_state(numpy_state)


class DDPMCDModelCompatibilityTests(unittest.TestCase):
    """Exercise the actual UNet, diffusion feature extraction and CD head."""

    scales = [2, 5, 8, 11, 14]
    timesteps = [1, 2, 3]

    @classmethod
    def setUpClass(cls):
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with isolated_seed(628):
            unet = UNet(
                in_channel=3, out_channel=3, inner_channel=4,
                norm_groups=1, channel_mults=[1, 2, 4, 8, 8],
                attn_res=[2], res_blocks=2, dropout=0.2, image_size=32,
            )
            cls.diffusion = GaussianDiffusion(
                unet, image_size=32, channels=3, loss_type="l2", conditional=False,
            ).eval()
            cls.diffusion.set_new_noise_schedule(
                {"schedule": "linear", "n_timestep": 10,
                 "linear_start": 1e-6, "linear_end": 1e-2},
                device=torch.device("cpu"),
            )
            cls.head = cd_head_v2(
                feat_scales=cls.scales.copy(), out_channels=2, inner_channel=4,
                channel_multiplier=[1, 2, 4, 8, 8], img_size=32,
                time_steps=cls.timesteps.copy(),
            ).eval()
        generator = torch.Generator(device="cpu").manual_seed(629)
        cls.before = torch.rand((1, 3, 32, 32), generator=generator) * 2 - 1
        cls.after = torch.rand((1, 3, 32, 32), generator=generator) * 2 - 1

    @classmethod
    def tearDownClass(cls):
        del cls.diffusion, cls.head
        torch.set_num_threads(cls._threads)

    def extract(self, seed):
        """Match upstream t-order, extracting A and then B independently."""
        features_a, features_b = [], []
        with isolated_seed(seed), torch.inference_mode():
            for timestep in self.timesteps:
                _, decoder_a = self.diffusion.feats(self.before, timestep)
                _, decoder_b = self.diffusion.feats(self.after, timestep)
                features_a.append(decoder_a)
                features_b.append(decoder_b)
        return features_a, features_b

    def test_small_real_unet_selected_decoder_shapes_and_logits(self):
        a, b = self.extract(630)
        expected = [(1, 4, 32, 32), (1, 8, 16, 16),
                    (1, 16, 8, 8), (1, 32, 4, 4), (1, 32, 2, 2)]
        self.assertEqual(len(a), len(self.timesteps))
        for stream in (a, b):
            for decoder in stream:
                self.assertEqual(len(decoder), 15)
                self.assertEqual([tuple(decoder[i].shape) for i in self.scales], expected)
        with torch.inference_mode():
            logits = self.head(a, b)
        self.assertEqual(tuple(logits.shape), (1, 2, 32, 32))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(set(logits.argmax(1).unique().tolist()).issubset({0, 1}))
        self.assertEqual(tuple(self.diffusion.betas.shape), (10,))
        self.assertEqual(len(self.diffusion.sqrt_alphas_cumprod_prev), 11)

    def test_pruning_unused_slots_is_bit_exact(self):
        a, b = self.extract(631)

        def selected_only(stream):
            return [[tensor if index in self.scales else None
                     for index, tensor in enumerate(decoder)] for decoder in stream]

        with torch.inference_mode():
            complete = self.head(a, b)
            pruned = self.head(selected_only(a), selected_only(b))
        torch.testing.assert_close(complete, pruned, rtol=0, atol=0)

    def test_seeded_feature_repeatability_and_rng_restoration(self):
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        first_a, first_b = self.extract(632)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

        second_a, second_b = self.extract(632)
        for left_stream, right_stream in ((first_a, second_a), (first_b, second_b)):
            for left, right in zip(left_stream, right_stream):
                for index in self.scales:
                    torch.testing.assert_close(left[index], right[index], rtol=0, atol=0)
        other_a, _ = self.extract(633)
        self.assertFalse(torch.equal(first_a[0][2], other_a[0][2]))

    def test_default_full_model_meta_structure_only(self):
        # No real full-model parameters/features are allocated on CPU or GPU.
        with torch.device("meta"):
            model = UNet(
                in_channel=3, out_channel=3, inner_channel=128,
                norm_groups=32, channel_mults=[1, 2, 4, 8, 8],
                attn_res=[16], res_blocks=2, dropout=0.2, image_size=256,
            )
            head = cd_head_v2(
                feat_scales=self.scales.copy(), out_channels=2, inner_channel=128,
                channel_multiplier=[1, 2, 4, 8, 8], img_size=256,
                time_steps=[50, 100, 400],
            )
            _, decoder = model(torch.empty(1, 3, 256, 256),
                               torch.empty(1, 1), feat_need=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 391048067)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 46405126)
        self.assertEqual([tuple(decoder[i].shape) for i in self.scales],
                         [(1, 128, 256, 256), (1, 256, 128, 128),
                          (1, 512, 64, 64), (1, 1024, 32, 32), (1, 1024, 16, 16)])


if __name__ == "__main__":
    unittest.main()
