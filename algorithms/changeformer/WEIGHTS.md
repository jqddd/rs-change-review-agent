# ChangeFormer 权重下载说明

本目录的模型权重 `checkpoints/best_ckpt.pt`（约 470MB）**没有提交到 git**（`.gitignore` 已忽略），需要自行下载后放到 `checkpoints/`。

## 下载方法

方式一（推荐，走 gh 的 SSH 通道较稳，国内 HTTPS 直连 release 常被重置）：

```bash
cd algorithms/changeformer
mkdir -p checkpoints
gh release download v0.1.0 -R wgcban/ChangeFormer -D checkpoints -p "*LEVIR*"
cd checkpoints && unzip -j -o "*.zip" "*/best_ckpt.pt" -d . && rm *.zip
```

方式二（直接 URL，若方式一失败再试）：

```bash
curl -L -o checkpoints/weight.zip "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip"
cd checkpoints && unzip -j -o "weight.zip" "*/best_ckpt.pt" -d . && rm weight.zip
```

## 权重信息

- 模型：ChangeFormerV6（`embed_dim=256`，二值建筑变化检测）
- 训练数据：LEVIR-CD
- 来源：[wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) 的 GitHub release v0.1.0（MIT 协议）
- 下载的 zip 内含 `best_ckpt.pt`（含模型权重 + 训练 optimizer 状态；推理只用其中的 `model_G_state_dict`）

## 单独验证算法

```bash
python inference.py --image_before 前时相.png --image_after 后时相.png
```
