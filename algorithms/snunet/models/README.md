# Upstream SNUNet-CD model

- Repository: https://github.com/likyoo/Siam-NestedUNet
- Commit: `aed3f1833f3d9d3eccba8c894fd097f358b5329e`
- Source: `models/Models.py` → `models/Models.py` (upstream layout retained).
- License: MIT; original notice is retained in `../LICENSE`.
- The model code is vendored without architectural changes. It depends only on PyTorch.

This module contains `SNUNet_ECAM` (SNUNet-CD, initial channels 32) and the upstream no-attention comparison model. Use the ECAM model for the official SNUNet-CD checkpoint.

The upstream test transform passes RGB values as float32 in the original 0–255 range, transposes HWC to CHW, and does not apply normalization; see `utils/transforms.py` at the same commit. The model returns a tuple containing two-channel logits; upstream `eval.py` selects the last output and applies class argmax.
