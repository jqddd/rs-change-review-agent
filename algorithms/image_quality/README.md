# 图像质量检查（无权重，CPU）

独立入口：`ImageQualityInspector(config=None).inspect(image_before, image_after)`。
输入严格为RGB8、256×256、PNG/JPEG；不做隐式颜色转换或缩放。

输出亮度/分位数、0与255端点占比、Laplacian方差、Tenengrad梯度能量、64×64分块指标及相对梯度线索。
黑白像素不推断为NoData；低纹理/常量的解释为unknown。
默认`uncalibrated_v1`阈值只产生review线索，不确诊模糊、云影，不是检测置信度。
输入假定全像素有效，没有接受valid-mask；不能将配准黑边当作正常图像质量样本。

工具：`tools/image_quality_tool.py:ImageQualityCheckTool`；完整运行说明见 `docs/review-tools.md`。

来源：原始组合实现，复用现有OpenCV/NumPy，无新增库或模型权重。
[OpenCV4.8梯度](https://docs.opencv.org/4.8.0/d5/d0f/tutorial_py_gradients.html)、
[直方图](https://docs.opencv.org/4.8.0/d1/db7/tutorial_py_histogram_begins.html)。
