# DDPM-CD 官方权重

官方来源：[README](https://github.com/wgcban/ddpm-cd#7-links-to-download-pre-trained-models)、
[GitHub initial_release](https://github.com/wgcban/ddpm-cd/releases/tag/initial_release)。
固定源码版本 `4970792f65227958ffaa1de787649ce2c5839f12`，MIT LICENSE。

## 本接入使用的配套组合

1. [diffusion-model-I190000_E97_gen.pth](https://github.com/wgcban/ddpm-cd/releases/download/initial_release/diffusion-model-I190000_E97_gen.pth)
   — 1,564,451,029 bytes，预训练扩散特征提取器。
2. [cd-levir-50-100-400-best_cd_model_gen.pth](https://github.com/wgcban/ddpm-cd/releases/download/initial_release/cd-levir-50-100-400-best_cd_model_gen.pth)
   — 185,642,343 bytes，LEVIR-CD 变化头，三个时间步 `[50,100,400]`。

在浏览器下载，保持文件名，放到：

```text
D:\AIProjects\Change-Agent\algorithms\ddpm_cd\checkpoints\
```

合计约1.75GB。官方 release 未提供 SHA256，文件大小检查只排除不完整下载，
不等于验证真实性；运行记录会保存本机计算的 SHA256。
未完成下载的 `.partial` 文件不能作为权重使用。
若浏览器添加 `(1)` 后缀，请提供实际路径或在完整下载后使用正确的文件名。

## 不要混用

- 两个文件都需要；扩散模型负责特征，检测头负责变化分类。
- `levir_test.upstream.jsonc` 原文件默认是 `[50,100]`，不能直接配三步权重。
  本接入用 `config/levir_50_100_400.json` 明确固定三步。
- 两步头、其他数据集的头、encoder特征或其他版本需要独立验证，首版不自动适配。
- 推理只读取裸 tensor state_dict `_gen.pth`；不需要 `_opt.pth` 优化器，也不需要完整模型 pickle 转换。
- 使用 `map_location=cpu`、`weights_only=True`、严格key/shape加载，不回退为随机权重。
- 大文件不提交Git；本算法 `.gitignore` 已排除整个 checkpoints 目录。

官方扩散预训练使用遥感影像，变化检测头训练域为 LEVIR-CD；其域外效果必须另测。
有公开权重不代表本接入已经复现论文分数，真实验证记录见 VALIDATION.md。
