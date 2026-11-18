# SNUNet-CD 独立工具接入

从公共基线 `eb54e664ab55c7eae6c69e3422de8ceeb620e4ec` 开发。
复用现有 PyTorch 2.1.2、NumPy、Pillow 和 Lagent，不安装或升级依赖。
算法源码只依赖 PyTorch；不需要 mmcv/mmseg 或对应打桩包。

## 文件关系

- `models/Models.py`：保留上游网络原文，附 MIT LICENSE 与来源说明。
- `inference.py`：独立推理，不依赖 Agent。
- `convert_checkpoint.py`：可信官方旧权重的一次性转换。
- `../../tools/snunet_tool.py`：输入校验、模型懒加载、运行记录与契约返回。
- `../../tools/demo_snunet_agent.py`：只注册 SNUNet 的 ReAct 示例，不启用 Python Interpreter。

所有本算法内部导入使用 `algorithms.snunet.*` 命名空间，不使用
`sys.path + from inference/models import ...`，避免与其他模型重名。

目录按搭档 ChangeFormer、BIT、TinyCD 的约定组织：

```text
algorithms/snunet/
    models/Models.py           # 模型实现，沿用官方文件名
    inference.py              # 独立推理入口
    checkpoints/              # 本地权重，不提交
    README.md
    WEIGHTS.md
    LICENSE
    convert_checkpoint.py     # SNUNet 旧模型格式专用转换
    VALIDATION.md             # 验证记录
tools/
    snunet_tool.py             # Lagent 工具适配
    demo_snunet_agent.py       # Agent 调用示例
tests/
    test_snunet.py
```

算法内部源码随模型而异，不复制其他模型的训练脚本或样例输出。
当前直接在原项目根目录 `D:\AIProjects\Change-Agent` 的 `feat/snunet-tool`
分支开发，`algorithms/`、`tools/` 与搭档保持相同层级。

## 权重与能力边界

先按 [WEIGHTS.md](WEIGHTS.md) 下载并转换官方 CDD 权重。
能力画像 `CAPABILITY` 位于工具代码中：二值、一般变化、CDD、256、CNN。
用户调研表中“SNUNet 无 LICENSE”已过时：所固定的官方版本包含 MIT LICENSE。
CDD 与 LEVIR-CD/LEVIR-MCI 数据域不同；本接入没有进行精度评估，也未训练模型。
`trained_on` 是部署者配置，不由 LLM 自行修改；与真实权重来源保持一致。

## 独立推理

在工作目录根目录执行，权重转换后才可获得有意义的预测。
示例中的输入为项目演示图片的相对路径；可替换为绝对路径。

```bat
python algorithms/snunet/inference.py --weight algorithms/snunet/checkpoints/snunet_cdd.pth --image_before Multi_change/demo_images/A.png --image_after Multi_change/demo_images/B.png --output output/snunet_label.png
```

也支持 `python -m algorithms.snunet.inference`；两种入口调用同一实现。

输出为单通道 0/1 PNG：0=未变化，1=变化。可视化或查看软件可能显示几乎全黑，
因为标签值1并不是255；工具会另存红色可视化图，不能用可视化代替计算标签。

## 工具直接调用

```python
from lagent.actions import ActionExecutor
from tools.snunet_tool import SNUNetTool

executor = ActionExecutor(actions=[SNUNetTool()])
result = executor('snunet', {
    'image_before': 'D:/path/to/A.png',
    'image_after': 'D:/path/to/B.png',
    'output_dir': 'output',
})
print(result.state, result.result, result.errmsg)
```

首版只接收 RGB、256×256、PNG/JPEG。对缺失文件、通道/尺寸不符返回
`ARGS_ERROR`，模型/权重加载、推理和写文件失败返回 `API_ERROR`。
成功返回契约 v0.1 信封、绝对文件路径、标签字典、像素坐标参考、变化像素数、
耗时、权重哈希、实际设备、来源与警告；每次生成独立目录与 `result.json`。
首调用耗时包括加载和权重哈希，后续调用复用模型；推理转CPU结果时完成GPU同步。
输入必须是完整有效像素瓦片；不支持地理量测、NoData 或局部裁剪回填。

## Agent 调用（需要你本机的 API Key）

```bat
python tools/demo_snunet_agent.py --weight algorithms/snunet/checkpoints/snunet_cdd.pth --question "请实际调用 snunet 工具，对比 D:/path/to/A.png 与 D:/path/to/B.png，将结果保存到 output，并用中文解释输出。"
```

未设置 `DEEPSEEK_API_KEY` 时会隐藏输入提示；不把Key写入文件。
可用 `--model` 和 `--endpoint` 指定服务配置。真实调用会产生 API 费用。
程序检查是否真的有成功的 SNUNet 执行记录；仅有LLM文字回答不算成功。
该检查只能确认发生了调用，不能保证摘要准确，也不是预算调度系统。

## 本地测试

```bat
python -m unittest discover -s tests -p test_snunet.py -v
```

测试区分：接口用受控假模型验证数据流和错误处理；网络烟雾测试使用临时随机参数
验证结构、预处理及严格加载。两者都不代表官方权重精度或真实LLM调用通过。
测试临时文件自动清理，不需要下载模型，不读取 API Key。

2026-11-18 发布前已额外验证本机官方 CDD 权重转换、独立推理与工具直接调用，
详见 [VALIDATION.md](VALIDATION.md)。尚未验证真实 DeepSeek 工具调用或完整测试集精度。
