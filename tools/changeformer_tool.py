"""ChangeFormer 工具适配（lagent BaseAction）。

把 ChangeFormer 算法包装成 Agent 可调用的独立工具，遵循 docs/tool-contract.md。
这是「工具适配层」：参数检查 → 调算法 → 整理文件引用与诊断 → 返回契约结果。
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Type

import numpy as np
from PIL import Image

from lagent.actions.base_action import BaseAction
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

# 让 `from inference import ChangeFormer` 找得到（算法实现层）
ALGO_DIR = Path(__file__).resolve().parents[1] / "algorithms" / "changeformer"
if str(ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(ALGO_DIR))

from inference import ChangeFormer  # noqa: E402

# 能力画像（docs/capability-profile.md v0.1 草案，供「多算法择优」调度使用）
CAPABILITY = {
    "output_encoding": "binary",      # 0=不变，1=变化
    "target_object": "building",
    "trained_on": "LEVIR-CD",
    "input_size": 256,
    "method_family": "transformer",
}


class ChangeFormerTool(BaseAction):
    """基于 Transformer 的双时相建筑变化检测，输出二值变化掩码（0=不变 / 1=变化）。"""

    __tool_name__ = "changeformer"

    def __init__(
        self,
        weight_path: Optional[str] = None,
        device: str = "cuda",
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
        enable: bool = True,
    ) -> None:
        super().__init__(description, parser, enable)
        self.weight_path = str(weight_path or (ALGO_DIR / "checkpoints" / "best_ckpt.pt"))
        self.device = device
        self._model: Optional[ChangeFormer] = None  # 懒加载，避免每次调用重复加载权重

    def _get_model(self) -> ChangeFormer:
        if self._model is None:
            self._model = ChangeFormer(self.weight_path, self.device)
        return self._model

    def run(self, image_before: str, image_after: str, output_dir: str = "output") -> ActionReturn | dict:
        """检测两张双时相影像之间的建筑变化。

        Args:
            image_before (str): 前时相影像路径（T1），PNG/JPG。
            image_after (str): 后时相影像路径（T2），PNG/JPG。
            output_dir (str): 输出目录，掩码保存到其下独立子目录。

        Returns:
            dict: 成功时返回符合契约的结果字典；失败时返回 ActionReturn。
        """
        # 契约第 4 节：调用前检查输入文件存在性
        for name, p in (("image_before", image_before), ("image_after", image_after)):
            if not Path(p).is_file():
                return ActionReturn(
                    type=self.name,
                    errmsg=f"{name} 文件不存在: {p}",
                    state=ActionStatusCode.ARGS_ERROR,
                )

        start = time.time()
        try:
            mask = self._get_model().detect(image_before, image_after)
        except Exception as exc:  # noqa: BLE001
            return ActionReturn(
                type=self.name,
                errmsg=f"推理失败: {exc}",
                state=ActionStatusCode.API_ERROR,
            )
        elapsed = time.time() - start

        # 契约 5.2：单通道整数标签文件 + 彩色可视化另存
        run_id = uuid.uuid4().hex[:12]
        out_dir = Path(output_dir) / f"changeformer_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        label_path = out_dir / "mask_label.png"
        viz_path = out_dir / "mask_viz.png"
        Image.fromarray(mask.astype(np.uint8)).save(label_path)  # 单通道 0/1 标签
        viz = np.zeros((*mask.shape, 3), dtype=np.uint8)
        viz[mask == 1] = [255, 0, 0]  # 变化区域标红
        Image.fromarray(viz).save(viz_path)

        changed_pixels = int((mask == 1).sum())

        # 契约第 6 节：统一结果信封
        return {
            "schema_version": "0.1",
            "run_id": run_id,
            "summary": f"检测到 {changed_pixels} 个变化像素",
            "artifacts": [
                {"kind": "mask_label", "path": str(label_path), "encoding": "0=不变,1=变化"},
                {"kind": "mask_viz", "path": str(viz_path)},
            ],
            "diagnostics": {"changed_pixel_count": changed_pixels},
            "warnings": ["ChangeFormer 为二值检测，不含道路/建筑类别区分"],
            "elapsed_seconds": round(elapsed, 2),
            "provenance": {
                "algorithm": "ChangeFormerV6",
                "trained_on": "LEVIR-CD",
                "weight": self.weight_path,
                "capability": CAPABILITY,
            },
        }
