# 模块4：图像预处理与修复

薄封装 OpenCV / scikit-image 的**现成实现**（非自研算法），为双时相变化检测提供
「修复」能力。对应 `docs/preprocessing-audit.md` 中结论：6 个检测算法都**不内置**
去噪/去模糊/亮度校正，这些由本模块补上。

- `histogram_match(reference, source)`：直方图匹配（辐射对齐），把 T2 亮度分布对齐到 T1。
- `denoise(image, method, ...)`：去噪，`method ∈ {nlmeans, bilateral, median, gaussian}`。
- `deblur(image, method, ...)`：去模糊/锐化（unsharp mask）。

所有函数输入/输出都是 `uint8` RGB HWC 数组，输出与检测工具契约兼容（0–255 uint8 RGB 256×256）。
本模块无权重、纯 CPU。
