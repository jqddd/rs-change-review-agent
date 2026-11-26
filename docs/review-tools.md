# 质量检查、错位检查与配准校正

本次新增三项独立工具及已有Lagent框架中的注册示例。沿用 `tool-contract.md`；没有修改共享Agent入口、训练环境或检测模型。

## 已启用的工具

- `image_quality_check`：`ImageQualityCheckTool`，输出全图/分块质量统计与review线索。采用Laplacian、Sobel、亮度和端点比例；不估计噪声标准差，不依赖尚未安装的PyWavelets。
- `alignment_check`：`AlignmentCheckTool`，phase相关估计整体平移，返回aligned/misaligned/unknown/refused。只生成诊断与可接受的候选变换，不校正原图。
- `registration_correct`：`RegistrationCorrectionTool`，phase初值＋translation ECC，返回registered/noop/unknown/refused。成功输出派生B、valid-mask、两方向矩阵、pair.json与完整运行记录。

名字由实例description明确固定；三者在 `tools/demo_review_tools_agent.py:make_executor` 中实例化并加入 `ActionExecutor`。这证明在该示例中已启用，不宣称其他旧Agent入口会自动拥有它们。

## 现有环境与运行

无需新增依赖、训练权重或GPU；使用当前Python3.9、OpenCV4.8、NumPy1.25和Pillow10。
从仓库根目录运行，输入为全有效RGB8 256×256 PNG/JPEG。本分支不包含本机demo图。
下面先用验证脚本生成一组可复现合成输入；实际使用时替换为自己的影像路径。
输出目录若已存在，换一个新目录，并同步替换后续输入路径。

```powershell
conda activate change-agent
python -B scripts/validate_review_tools.py --output outputs/review-tools-example
python -B -m tools.demo_review_tools_agent --smoke --image-before outputs/review-tools-example/inputs/integer_positive_A.png --image-after outputs/review-tools-example/inputs/integer_positive_B.png --output-dir outputs/review_tools
```

`--smoke`实际调用三项工具并打印返回记录，不联网、不代表LLM自主选择。
可以直接调用executor：

```python
import json
from tools.demo_review_tools_agent import make_executor

executor = make_executor()
action = executor('alignment_check', json.dumps({
    'image_before': 'outputs/review-tools-example/inputs/integer_positive_A.png',
    'image_after': 'outputs/review-tools-example/inputs/integer_positive_B.png',
    'output_dir': 'outputs/review_tools',
}))
print(action.state, action.result, action.errmsg)
```

## DeepSeek入口

```powershell
python -B -m tools.demo_review_tools_agent --model deepseek-flash --question "请实际依次调用 image_quality_check、alignment_check、registration_correct。image_before=outputs/review-tools-example/inputs/integer_positive_A.png，image_after=outputs/review-tools-example/inputs/integer_positive_B.png，output_dir=outputs/review_tools。根据实际状态回答；unknown/refused不得声称校正成功；输出存在无效边界时说明不能直接交给现有全有效256检测器。"
```

模型名和endpoint可按已可用账户配置调整。优先读`DEEPSEEK_API_KEY`，否则隐藏输入，不保存密钥。只有实际成功调用工具才打印VERIFIED；诊断unknown/refused本身仍可能是正常SUCCESS执行，必须阅读diagnostics。
执行器强制每个实例最多6次工具调用、其中最多2次校正，ReAct最多8轮；没有启用通用Python Interpreter。
仅接受声明的独立工具名；点号API别名不执行，避免绕过调用预算或调用未声明方法。
这不是完成了自主调度策略，调用预算也不是配准精度保证。

## 输出与安全边界

每次新建独立run目录，返回绝对路径、原图hash、源码hash、配置、无权重说明和CPU耗时。
质量原始分块指标保存在`quality_metrics.json`，两张梯度图共用log1p显示范围，**仅是量测可视化，不是坏像素Mask**。
阈值为`uncalibrated_v1`/`uncalibrated_translation_v1`，只能触发review，不提供校准confidence。

固定T1参考网格。校正图使用一次双线性重采样，`valid_mask.png`为单通道0/1，标明可用采样区，不能把黑边当拆除。
存在无效边缘时，派生PNG还声明`NoData_mask`路径；这不是把某个RGB值当作NoData，新工具会拒绝将其重新当作全有效原图。
`pair.json`含downstream条件：非全有效时，`compatible_with_existing_full_valid_256_detectors=false`。
当前SNUNet/DDPM等要求全有效RGB256且不接收valid-mask，所以本入口**不自动把校正结果交给检测器**。后续需要显式设计有效区域、裁剪/尺寸适配和回填关系，再单独验证下游性能；不静默resize。
质量检查本身也不接收valid-mask，应先对原始全有效图检查。

参数、输入条件错误=ARGS_ERROR，实际计算/写入故障=API_ERROR。
正常计算发现未知/拒绝校正仍为SUCCESS并返回真实诊断；不会写出伪成功图像。
新工具局部使用object-only JSON parser，拒绝标量/数组/代码文本；没有改变原Interpreter的解析方式。

## 验证

```powershell
python -B -m unittest discover -s tests -p "test_image_quality.py" -v
python -B -m unittest discover -s tests -p "test_registration.py" -v
python -B -m unittest discover -s tests -p "test_review_tools.py" -v
python -B scripts/validate_review_tools.py
```

验证脚本使用独立输出目录，存在则拒绝覆盖；换目录通过`--output`指定。
包含正常/no-op、正负整数/亚像素平移、局部真变化、常量、无关输入、过大位移和模糊等场景，及本机已提取遥感底图的已知合成位移。
扰动真值只由离线评估读取，不传给工具/Agent。真实遥感底图＋合成已知变换可以验证实现几何方向，不能当作真实双时相配准的独立检查点验收。
单元测试中scripted LLM经过真实ReAct与ActionExecutor执行工具，但只证明协议链路；真实DeepSeek调用、自主路由收益和真人同伴验收须分别记录，不能用测试替代。

## 2026-11-23 本机验证记录

- 45项CPU测试通过：质量17项、配准14项、工具与执行器14项；包含非法JSON、私有API、预算绕过、后置PNG NoData、I/O错误和无效边界处理。
- 10类合成场景通过；no-op保持原像素，局部真变化没有被平移模型吞掉，证据不足时不生成校正图。
- 20张遥感底图各施加已知正/负平移，20张均接受校正；已知位移恢复误差中位数0.007795像素、最大0.019053像素。这是人工已知几何实现验证，不是真实双时相独立检查点TRE或真实场景精度保证。
- 最终原始记录：`outputs/review-tools-validation-20261123-verified/validation.json`，图示：同目录`registration_evidence.png`。每次工具调用的独立JSON、矩阵、派生图和valid-mask保存在其`tool_runs`子目录。
- 当前demo_images/A.png与B.png实际执行时，质量工具返回量测，错位/校正工具因phase证据不足返回unknown，未强行校正。
- 已完成独立代码审查。当前环境未提供`DEEPSEEK_API_KEY`，没有真实DeepSeek调用验收；示例支持隐藏输入，密钥不会写入文件。自主选择策略、检测器有效区衔接和真人同伴验收仍需分别完成。
