# DDPM-CD 独立算法与 Agent 工具

基于公共基线 `eb54e664ab55c7eae6c69e3422de8ceeb620e4ec`，功能分支
`feat/ddpm-cd-tool`。目录组织与 ChangeFormer、BIT、TinyCD、SNUNet 接入一致。

官方仓库：<https://github.com/wgcban/ddpm-cd>，固定版本
`4970792f65227958ffaa1de787649ce2c5839f12`，MIT LICENSE 随源码保留。

## 文件结构

```text
algorithms/ddpm_cd/
    models/sr3_modules/         # 官方扩散 UNet 与特征提取
    models/cd_modules/          # 官方变化检测头
    config/levir_50_100_400.json # 本接入明确使用的三时间步配置
    config/levir_test.upstream.jsonc # 官方原始配置参考（默认两时间步）
    inference.py               # 独立推理，不依赖 Lagent
    checkpoints/               # 两组本地权重，不提交
    WEIGHTS.md
    VALIDATION.md
    LICENSE
    upstream_manifest.json     # 来源、固定版本与源码适配说明
tools/
    ddpm_cd_tool.py            # 独立 Lagent BaseAction
    demo_ddpm_cd_agent.py      # 仅启用 DDPM-CD 的 ReAct 示例
tests/
    test_ddpm_cd_models.py
    test_ddpm_cd_tool.py
```

正式推理只使用最小官方网络模块；不执行原训练入口、WandB、性能分析器、
图像生成或数据集加载脚本。所有导入使用 `algorithms.ddpm_cd.*` 命名空间，
避免与其他算法的 `model/models/inference` 重名。

## 必需的两组权重

按 [WEIGHTS.md](WEIGHTS.md) 下载扩散特征提取器和匹配的三时间步 LEVIR-CD
变化检测头。没有任何一组权重时都会明确失败，不会使用随机模型冒充预测。
优化器状态不用于推理，无需下载 `_opt.pth`。

本接入首版只声明支持这套 LEVIR-CD、decoder、`t=[50,100,400]` 的权重与配置。
其他训练集、时间步、编码器特征或模型版本需要独立的部署配置与验证；
不能仅凭网络参数能够加载，就给它标上相同的训练域或适用性。

## 环境

复用现有 Python 3.9、PyTorch 2.1.2+cu121、NumPy、Pillow、tqdm、Lagent。
不需安装原仓库 Linux 专用环境、WandB、lmdb、tensorboardX 或 thop。
使用 PyTorch 2.1 的 meta 初始化、`assign=True` 严格加载和可用时的 mmap，
减少大权重在 CPU 内存中的重复副本。权重在加载期间保持不可变。
CUDA 不可用时不会静默换设备，可明确指定 `--device cpu`。

实际模型较大，8GB 显存是否足够需要真实权重 batch=1 测量；结构测试不能证明。
首次加载、权重哈希和稳态推理成本在记录中区分。

## 独立推理

在项目根目录，激活 `change-agent` 环境后运行：

```bat
python algorithms/ddpm_cd/inference.py --image_before Multi_change/demo_images/A.png --image_after Multi_change/demo_images/B.png --output algorithms/ddpm_cd/output/demo_label.png
```

也支持 `python -m algorithms.ddpm_cd.inference`。
权重可通过 `--diffusion-weight`、`--change-weight` 指定，部署随机种子通过
`--seed` 指定；这些配置不作为 LLM 可自由修改的工具参数。
输入必须是已对齐、无无效区域的 RGB 256×256 PNG/JPEG，不静默缩放或交换时序。
输出标签为单通道 uint8：`0=未变化，1=建筑变化`，不区分道路或新增/拆除方向。
单通道标签值1看起来接近黑色，不能据此判断空预测；工具会另存红色可视化。

## 工具调用

```python
from lagent.actions import ActionExecutor
from tools.ddpm_cd_tool import DDPMCDTool

executor = ActionExecutor(actions=[DDPMCDTool()])
result = executor('ddpm_cd', {
    'image_before': 'D:/path/to/A.png',
    'image_after': 'D:/path/to/B.png',
    'output_dir': 'algorithms/ddpm_cd/output',
})
print(result.state, result.result, result.errmsg)
```

与其他工具共同启用时，把实例加入同一个 ActionExecutor；仅有文件或类注册不算启用。
工具返回契约 v0.1 信封、绝对路径、标签、像素坐标参考、变化像素数、两组权重哈希、
实际配置/随机种子/设备/耗时，并保存 `result.json`。
参数不满足返回 `ARGS_ERROR`；缺失/不完整权重、推理或写文件失败返回 `API_ERROR`。
它不执行配准或质量复核，不虚构置信度、平方米或建筑栋数。

## Agent 示例

```bat
python tools/demo_ddpm_cd_agent.py --question "请实际调用 ddpm_cd，对比 D:/AIProjects/Change-Agent/Multi_change/demo_images/A.png 和 D:/AIProjects/Change-Agent/Multi_change/demo_images/B.png，结果保存在 algorithms/ddpm_cd/output，并用中文解释。"
```

未设置 `DEEPSEEK_API_KEY` 时，通过隐藏输入读取 Key，不保存到文件。
可用 `--model`、`--endpoint` 调整 API 配置。权重先加载成功，才进行付费 API 调用。
如果没有成功执行 DDPM-CD 的工具记录，脚本会报告未验证，不把 LLM 编写的文字当成结果。
这不等于已实现自主择优、预算调度或事实核验。

## 与官方推理的一致性

- PIL RGB 对应 float32 CHW，`/255 * 2 - 1`；没有标签输入，也没有测试增强。
- 线性2000步噪声调度，beta `1e-6 → 1e-2`；恢复权重后设置官方测试调度。
- 按 `[50,100,400]` 顺序，每个时间步先提取 T1、再提取 T2 的 decoder 特征。
- 使用尺度 `[2,5,8,11,14]`，保留原15槽位置；释放未使用特征不改变检测头输入。
- 两个模型均 eval，推理不求梯度；保留官方检测头前向的细节，没有修改网络算法。
- 这是六次特征前向和一次检测头计算，不是2000步逆扩散图像生成。
- 官方测试阶段也随机采样噪声。这里固定部署 seed（默认0），保留独立 A/B/t 随机采样，
  执行后恢复 NumPy、CPU Torch 和当前 GPU RNG 状态。串行复核更可重复，但不同种子仍需
  在正式评估中测量波动；不保证跨硬件/软件版本逐位一致。其他线程若同时使用全局 RNG，
  需由公共调度层协调。

## 验证范围

```bat
python -m unittest discover -s tests -p "test_ddpm_cd*.py" -v
```

12项离线测试已通过，包括小型真实网络特征与检测头、full/pruned特征结果一致、
完整默认网络的meta结构、随机数恢复、双权重接口、输入校验和ReAct反馈。
假检测器、小型随机网络与meta测试均不代表预训练模型精度、真实8GB显存或DeepSeek调用已通过。
真实权重与 API 验收状态以 [VALIDATION.md](VALIDATION.md) 为准。
