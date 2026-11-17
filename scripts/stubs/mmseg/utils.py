"""最小化的 mmseg.utils.get_root_logger 替身。"""
from __future__ import annotations

import logging


def get_root_logger(log_file: str | None = None, log_level: int = logging.INFO) -> logging.Logger:
    """返回一个带默认 handler 的 root logger（对齐 mmseg.utils.get_root_logger）。"""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    return logger
