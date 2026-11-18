# SNUNet-CD 权重

官方仓库：<https://github.com/likyoo/Siam-NestedUNet>，固定代码版本
`aed3f1833f3d9d3eccba8c894fd097f358b5329e`，MIT 许可证。

## 目前确认的官方权重

官方 README 提供的是 **CDD** 权重 `snunet-32.pt`：

- [百度网盘](https://pan.baidu.com/s/1X10jbpzywt1tslkuTm03rg?pwd=ki1u)，提取码 `ki1u`。
- [Google Drive 文件夹](https://drive.google.com/drive/folders/1_aoUvMC8zWy4Pv7vU_BHu5yzzjgIsNXv?usp=sharing)。

截至接入时，尚未确认可下载的官方 LEVIR-CD 权重。不能把 CDD 权重标为
LEVIR-CD 模型，也不能保证它在建筑变化测试集上的精度。
CDD 模型输出一般二值变化，不提供道路/建筑类别或新增/拆除方向。

权重不随代码提交。下载后放在 `algorithms/snunet/checkpoints/snunet-32.pt`。
`checkpoints/` 已在本算法目录的 `.gitignore` 中排除。

## 一次性转换

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
若你已有同结构纯 state_dict，可以在推理时直接指定其路径，无需转换。
第三方镜像权重需要独立核实来源，不应仅凭同名文件标作官方。

## 预处理依据

- 官方 `utils/transforms.py`：测试时不 Normalize，RGB 原始 0–255 转 float32 CHW。
- 官方 `eval.py`：使用最后一个输出，再按类别 argmax。
- 本接入首版只接受完整、已对齐的 RGB 256×256 PNG/JPEG；不静默缩放或转换通道。
- 对齐质量与遮挡等由上游诊断工具负责；SNUNet 本身不验证配准。

样例推理不能代替完整测试集评估；实际验证范围见 [VALIDATION.md](VALIDATION.md)。
