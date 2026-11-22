# Upstream SNUNet-CD model

## Default Open-CD LEVIR C16

`opencd_c16.py` retains the Open-CD backbone numerical implementation and adds
the configured FCNHead inference path. Registry-only code and the global
`dump_patches` toggle are omitted. The C16 backbone and classifier preserve
the checkpoint's `backbone.*` and `decode_head.conv_seg.*` names.

Source commit: `790a1972b538baaaffd4c3022a4d9afaa3f861e5`.
Apache-2.0 and the original notice are retained in `../LICENSE.opencd`;
exact source hashes, checkpoint/config references and adaptations are in
`../opencd_source.json`. This source revision is not asserted to be the
unrecorded training-time code revision.

PIL inputs are already RGB. The deployment subtracts the raw-pixel RGB mean
and divides by RGB standard deviation, with no /255 or second channel swap.
It uses native 256 tiles; upstream published testing uses whole 1024 images.

## Explicit legacy CDD C32

- Repository: https://github.com/likyoo/Siam-NestedUNet
- Commit: `aed3f1833f3d9d3eccba8c894fd097f358b5329e`
- Source: `models/Models.py` → `models/Models.py` (upstream layout retained).
- License: MIT; original notice is retained in `../LICENSE`.
- The model code is vendored without architectural changes. It depends only on PyTorch.

This module contains `SNUNet_ECAM` (SNUNet-CD, initial channels 32) and the upstream no-attention comparison model. Use the ECAM model for the official SNUNet-CD checkpoint.

The upstream test transform passes RGB values as float32 in the original 0–255 range, transposes HWC to CHW, and does not apply normalization; see `utils/transforms.py` at the same commit. The model returns a tuple containing two-channel logits; upstream `eval.py` selects the last output and applies class argmax.
