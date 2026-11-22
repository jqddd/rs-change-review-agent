# SNUNet-CD 独立算法与工具

默认部署已切换为 **Open-CD 的 LEVIR-CD C16 建筑变化权重**。
旧 CDD C32 仍通过显式 `--profile cdd_c32` 使用，权重和历史实验记录保留。
两个 profile 绑定各自的网络、分类头、预处理和标签，不能只替换权重文件名。

开发基线为 `eb54e664ab55c7eae6c69e3422de8ceeb620e4ec`，分支 `feat/snunet-tool`。
复用现有 Python 3.9、PyTorch 2.1.2+cu121、NumPy、Pillow 和 Lagent，
未安装或升级 mmcv、mmsegmentation、mmengine 等公共依赖。

## 文件结构

```text
algorithms/snunet/
    models/Models.py               # 原 CDD C32，保留原文及 MIT 许可
    models/opencd_c16.py           # Open-CD C16 + FCN 分类头的纯 PyTorch 推理
    inference.py                  # 默认 levir_c16；可显式选 cdd_c32
    convert_opencd_checkpoint.py  # 静态提取固定 Open-CD 权重，不执行 pickle 对象
    convert_checkpoint.py         # 旧 CDD 全模型对象专用转换
    checkpoints/                  # 本地权重，忽略提交
    README.md
    WEIGHTS.md
    VALIDATION.md
tools/
    snunet_tool.py                # 独立 BaseAction，统一结果信封
    demo_snunet_agent.py          # 单工具 ReAct 示例
tests/
    test_snunet.py                # 原接口与 CDD 兼容回归
    test_snunet_c16.py            # C16 配置、严格加载及安全转换回归
```

内部使用 `algorithms.snunet.*` 命名空间，算法可脱离 Lagent 单独运行。

## 默认权重与配置

作者模型库的原文件为 `snunet_c16_256x256_40k_levircd.pth`，
转换后的默认文件为 `checkpoints/snunet_levir_c16.pth`。获取及转换命令见
[WEIGHTS.md](WEIGHTS.md)。日常加载仅使用 `weights_only=True`，并严格检查参数。

C16 使用 RGB 原始0–255值减均值 `[123.675,116.28,103.53]`、
除标准差 `[58.395,57.12,57.375]`；不先除255，也不对 PIL RGB 再交换颜色通道。
输出为 `0=未变化，1=建筑变化`，不区分新增/拆除，不包含道路类别。

本工具继续接收已对齐、完整有效的 RGB 256×256 PNG/JPEG 原生瓦片，
不静默缩放。Open-CD 原配置的测试流程使用1024×1024整图，
因此当前瓦片部署不等同于原公开测试协议，不能直接引用其论文/模型库分数。

## 独立推理

在项目根目录和 `change-agent` 环境中执行：

```bat
python algorithms/snunet/inference.py --image_before Multi_change/demo_images/A.png --image_after Multi_change/demo_images/B.png --output algorithms/snunet/output/demo_c16_label.png
```

也支持 `python -m algorithms.snunet.inference`。
可通过 `--weight` 指定同一 profile 的转换权重，`--device cpu` 显式使用CPU。
已有输出文件会拒绝覆盖。标签值1在看图软件中接近黑色，不代表空预测。

旧版本的显式调用：

```bat
python algorithms/snunet/inference.py --profile cdd_c32 --weight algorithms/snunet/checkpoints/snunet_cdd.pth --image_before Multi_change/demo_images/A.png --image_after Multi_change/demo_images/B.png --output algorithms/snunet/output/demo_cdd_label.png
```

## 工具调用

```python
from lagent.actions import ActionExecutor
from tools.snunet_tool import SNUNetTool

executor = ActionExecutor(actions=[SNUNetTool()])  # 默认 LEVIR C16
result = executor('snunet', {
    'image_before': 'D:/path/to/A.png',
    'image_after': 'D:/path/to/B.png',
    'output_dir': 'algorithms/snunet/output',
})
print(result.state, result.result, result.errmsg)
```

模型懒加载并复用。工具返回单通道标签、红色可视化、`result.json`、
profile、能力画像、实际预处理、网络配置、权重哈希及耗时。
缺失/错误尺寸/错误通道返回 `ARGS_ERROR`，模型、权重或写文件失败返回 `API_ERROR`。
无地理参考时只报告像素数，不报告平方米或建筑栋数。

部署参数 `profile` 不开放给LLM。旧版可构造 `SNUNetTool(profile='cdd_c32')`；
能力画像和工具说明随profile切换。`trained_on` 如显式传入，必须与profile相符，
不能把CDD模型只改一个字符串就标为LEVIR模型。

## Agent 示例

```bat
python tools/demo_snunet_agent.py --question "请实际调用 snunet，对比 D:/AIProjects/Change-Agent/Multi_change/demo_images/A.png 与 D:/AIProjects/Change-Agent/Multi_change/demo_images/B.png，结果保存在 algorithms/snunet/output，并用中文回答。"
```

默认先加载C16，成功后才访问API。Key通过环境变量或隐藏输入读取，不保存到文件。
日志必须包含真实成功的工具记录；仅有LLM文字不算执行成功。
该示例只启用一个SNUNet工具，不代表多算法自主调度已完成。

## 验证与旧结果边界

```bat
python -m unittest discover -s tests -p "test_snunet*.py" -v
```

测试与真实权重运行记录见 [VALIDATION.md](VALIDATION.md)。
旧CDD C32在LEVIR-MCI建筑任务上的IoU 8.73%、F1 16.06%只属于旧profile，
不能作为新C16效果证据，也不能据此否定SNUNet架构。
对新profile的目标域评估须另行记录，不能用样例变化像素数证明精度。
