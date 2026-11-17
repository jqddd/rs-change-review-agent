# BIT 权重下载说明

权重 `checkpoints/best_ckpt.pt` **没有提交到 git**（`.gitignore` 已忽略），需自行下载后放到 `checkpoints/`。

## 下载方法

- 百度网盘：https://pan.baidu.com/s/1HiXwpspl6odYQKda6pMuZQ ，提取码：`2lyz`
- Google Drive：https://drive.google.com/file/d/1IVdF5a3e1_7DiSndtMkhpZuCSgDLLFcg/view

下载后把 `best_ckpt.pt` 放到本目录的 `checkpoints/best_ckpt.pt`。

## 权重信息

- 模型：BIT（`base_transformer_pos_s4_dd8_dedim8`，二值建筑变化检测）
- 训练数据：LEVIR-CD
- 来源：[justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD)
- ⚠️ **许可协议：该仓库没有 LICENSE 文件**（未声明开源协议），商用或再分发前需自行确认授权

## 单独验证算法

```bash
python inference.py --image_before 前时相.png --image_after 后时相.png
```
