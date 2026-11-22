# SNUNet-CD 权重

## 默认 Open-CD LEVIR C16

2026-11-22默认部署切换为作者维护的Open-CD LEVIR-CD C16。
来源：[作者模型库](https://huggingface.co/likyoo/Open-CD_Model_Zoo/tree/main/LEVIR-CD)。

- 原文件：`snunet_c16_256x256_40k_levircd.pth`，40,792,765 bytes。
- 固定发布版本：`07de9dd7027827cb47bdbe9bcece7ff9b8da73ad`。
- 模型库LFS SHA256：`984bfd07e0a30bbea5cd5a2fec918c994208c63b6c69e67a6592e06871811572`。
- [固定版本下载](https://huggingface.co/likyoo/Open-CD_Model_Zoo/resolve/07de9dd7027827cb47bdbe9bcece7ff9b8da73ad/LEVIR-CD/snunet_c16_256x256_40k_levircd.pth?download=true)。

放入 `algorithms/snunet/checkpoints/` 后，一次性转换：

```bat
python algorithms/snunet/convert_opencd_checkpoint.py --source algorithms/snunet/checkpoints/snunet_c16_256x256_40k_levircd.pth --output algorithms/snunet/checkpoints/snunet_levir_c16.pth
```

原文件还包含训练日志、优化器及MMEngine对象。转换器固定检查文件大小和SHA，
使用pickletools把序列化指令解释为惰性数据，仅读取ZIP中的tensor storage，
不导入或执行pickle对象，不需要安装MMEngine。日常推理仅读取转换后的纯tensor与基础元数据，
并要求 `profile=levir_c16` 和完整严格state_dict加载。

原生训练/验证配置字符串保存在转换文件中，不会作为Python执行。
网络为16基础通道SNUNet_ECAM加独立FCN分类头；RGB减均值、除标准差。
当前工具使用原生256瓦片，原公开测试使用1024整图；发布分数不自动适用于本部署。
这份权重不能由原CDD C32实现直接加载，原CDD的池化修复也不会套用到C16网络。

## 保留的旧 CDD C32

官方仓库：<https://github.com/likyoo/Siam-NestedUNet>，固定代码版本
`aed3f1833f3d9d3eccba8c894fd097f358b5329e`，MIT 许可证。

### 原仓库 CDD 权重

官方 README 提供的是 **CDD** 权重 `snunet-32.pt`：

- [百度网盘](https://pan.baidu.com/s/1X10jbpzywt1tslkuTm03rg?pwd=ki1u)，提取码 `ki1u`。
- [Google Drive 文件夹](https://drive.google.com/drive/folders/1_aoUvMC8zWy4Pv7vU_BHu5yzzjgIsNXv?usp=sharing)。

原接入只核实了CDD；现已补充上方Open-CD的LEVIR-CD权重。
不能把旧CDD权重标为LEVIR-CD模型，也不能保证它在建筑变化测试集上的精度。
CDD 模型输出一般二值变化，不提供道路/建筑类别或新增/拆除方向。

权重不随代码提交。下载后放在 `algorithms/snunet/checkpoints/snunet-32.pt`。
`checkpoints/` 已在本算法目录的 `.gitignore` 中排除。

### 旧 CDD 一次性转换

官方 `train.py` 保存整个模型对象，旧类路径为 `models.Models.*`。
日常推理只加载纯 tensor state_dict，避免在进程中注入全局 `models` 别名。
仅对从可信来源取得的官方模型执行以下转换：旧 pickle 可以执行代码。

在 SNUNet 工作目录的根目录，使用已验证的 `change-agent` 环境：

```bat
python algorithms/snunet/convert_checkpoint.py --source algorithms/snunet/checkpoints/snunet-32.pt --output algorithms/snunet/checkpoints/snunet_cdd.pth --trust-source
```

转换程序严格检查模型结构，并打印原文件 SHA256；它不训练，也不修改参与推理的参数。
官方 CDD 旧权重额外保存了 `ca2/ca3/ca4` 的六个张量，但文件内嵌的原始
`forward` 与固定上游版本的 `forward` 都只使用 `ca/ca1`。转换时把这六个
未使用的张量保存在 `unused_legacy_state` 字段，推理用 `state_dict` 仍严格加载，
其他未知参数、缺失参数或形状不匹配不会被静默忽略。

转换还必须保存无参数的池化配置：官方旧 CDD 完整模型的 `ca/ca1.max_pool`
虽然名称含 max，实际对象是 `AdaptiveAvgPool2d(1)`，而当前上游源码改成了
`AdaptiveMaxPool2d(1)`。只保存 state_dict 会漏掉这一差异并改变预测。
本转换器保存 `model_configuration.attention_pooling`，推理时恢复它；
旧版转换产物如果缺少这份配置会明确拒绝加载，需要从原始 `.pt` 重新转换。
已增加原完整模型与转换重建模型的前向输出一致性回归测试。
旧CDD的调用需显式传入 `--profile cdd_c32`。
若你已有同结构纯 state_dict，可以在该profile中直接指定其路径，无需转换。
第三方镜像权重需要独立核实来源，不应仅凭同名文件标作官方。

### 旧 CDD 预处理依据

- 官方 `utils/transforms.py`：测试时不 Normalize，RGB 原始 0–255 转 float32 CHW。
- 官方 `eval.py`：使用最后一个输出，再按类别 argmax。
- 本接入首版只接受完整、已对齐的 RGB 256×256 PNG/JPEG；不静默缩放或转换通道。
- 对齐质量与遮挡等由上游诊断工具负责；SNUNet 本身不验证配准。

样例推理不能代替完整测试集评估；实际验证范围见 [VALIDATION.md](VALIDATION.md)。
