"""最小化的 mmcv.runner.load_checkpoint 替身。

Change-Agent 的 segformer.py 只在训练时调用它（init_weights），推理不调用。
这里给一个「能真正干活」的版本，签名对齐 mmcv，避免将来误用时出错。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

import torch


def load_checkpoint(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    map_location: Union[str, Any, None] = None,
    strict: bool = False,
    logger: Optional[logging.Logger] = None,
    revise_keys: Optional[list] = None,
) -> dict:
    """加载 checkpoint 到 model（对齐 mmcv.runner.load_checkpoint 的签名）。"""
    if logger is None:
        logger = logging.getLogger(__name__)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at {filename}")
    checkpoint = torch.load(filename, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Checkpoint format not supported")
    if revise_keys is not None:
        raise NotImplementedError("revise_keys not supported in this stub")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded checkpoint from {filename}")
    return checkpoint
