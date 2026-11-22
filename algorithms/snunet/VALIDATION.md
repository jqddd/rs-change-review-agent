# SNUNet 验证记录

## 默认部署切换为 Open-CD LEVIR C16 2026年11月22日

默认profile为 `levir_c16`，旧CDD仍可通过 `cdd_c32` 显式运行。
网络、分类头、权重、预处理、能力画像和结果标签均按profile绑定；未升级现有依赖。

### 来源和实现一致性

- 作者Open-CD模型库原权重40,792,765bytes，SHA256与LFS记录一致：
  `984bfd07e0a30bbea5cd5a2fec918c994208c63b6c69e67a6592e06871811572`。
- 安全静态提取236个tensor，共3,015,856个元素；未执行原checkpoint中的MMEngine/NumPy pickle对象。
- 正式转换文件 `checkpoints/snunet_levir_c16.pth` 的本机SHA256为
  `314d74b0b16261e6d0917b995673de660c7b13e02e475aeec019dca9e98ef671`。
- 对真实权重严格加载；16基础通道、64→2 FCN分类头、RGB mean/std、BN和注意力池化均核对。
- 使用原始下载的官方backbone与FCNHead eval的1×1卷积作为独立网络前向参照，
  在CPU 64×64真实权重输入上与本接入logits最大绝对差为0。
  这验证网络和分类头推理等价，不是原始1024测试程序的完整复现。
- 固定源码、权重、配置、许可和适配差异记录见 `opencd_source.json`。

### 代码及真实工具验收

`python -m unittest discover -s tests -p "test_snunet*.py" -v`：20项CPU测试通过。
包括原CDD兼容、C16结构与归一化、严格加载和profile隔离、工具标签及运行记录、
静态转换的惰性对象处理、storage边界、错误状态和输出保护。

本机RTX 4060 Laptop、PyTorch2.1.2+cu121、FP32、batch=1真实运行：

- 演示A/B输出22,375个建筑变化像素，A/A为0；
  LEVIR-MCI test_000001为0，test_000004为12,909。
- 独立预测与真实ActionExecutor调用的Mask逐位一致；状态SUCCESS，模型复用，
  0/1单通道标签、红色可视化及result.json正常保存；缺失输入为ARGS_ERROR。
- 默认profile的真实CLI输出与独立API一致；推理、Agent示例、C16转换器的
  直接脚本与 `python -m` 共6种帮助入口通过，直接脚本也从项目外工作目录验证。
  显式CDD C32在GPU演示A/B上仍输出11,694个变化像素，与旧记录一致。
- 当前推理已分配显存峰值约105.2MiB，reserved峰值158MiB。
  该数值不是nvidia-smi全设备用量，也不是多模型同时驻留测试。

### 完整LEVIR-MCI建筑任务适用性评估

沿用此前官方test.txt的全部1,929对原生RGB256瓦片：建筑正样本878、无标注变化953、
仅道路变化98。以原标签第一通道255作为建筑正类，道路128和背景0作为负类。
累计整个测试集的TP/FP/FN/TN，未训练、调阈值、交换时序、缩放或测试增强；
真值只由独立评估代码读取，未提供给模型或Agent。

- C16：建筑前景IoU **82.3912%**，F1 **90.3456%**，
  Precision **91.2921%**，Recall **89.4185%**。
- TP6,052,277，FP577,298，FN716,206，TN119,073,163。
- 已重新汇总1,929行CSV，确认与汇总JSON计数和指标一致。
- 当前接口平均约48.96ms/对，GPU同步、3次预热，含文件校验/预处理/推理和CPU结果，
  不含首次加载、标注和指标计算。CPU工具与并发环境有差别，不作为纯网络速度排名。
- 此前固定参考：CDD C32 IoU8.7313%、F116.0603%；MCI建筑类别IoU81.9999%、F190.1098%。
  C16与旧CDD是两套部署方案，不能把差异只归因于训练数据。

这些是LEVIR-MCI建筑前景适用性结果，不是Open-CD公开LEVIR-CD整体测试分数、
MCI论文多类别MIoU或独立域外泛化结果。LEVIR-CD训练与本地LEVIR-MCI的地理/样本重叠
尚未排除，不能据此宣称无泄漏泛化或Agent调度收益。

本地原始产物：`output/c16_validation_20261122_f26501af/validation.json`、
`c16_per_image.csv`、`c16_predictions.npz`、`examples.png`和独立工具结果目录。
验收脚本：根目录 `work/validate_c16_20261122.py`；网络对照记录
`work/c16-research/reference_parity.json`。权重、输入和运行产物不提交Git。

真实DeepSeek调用、与其他模型联合调度、独立域外验证及复核收益仍未验收。

## 历史记录 2026年11月17日

环境：Windows、Python 3.9、PyTorch 2.1.2+cu121、RTX 4060 Laptop GPU。
未安装或升级公共环境依赖。开发分支为 `feat/snunet-tool`，基于公共基线
`eb54e664ab55c7eae6c69e3422de8ceeb620e4ec`。

执行 `python -m unittest discover -s tests -p test_snunet.py -v`，六项全部通过。
测试覆盖：

1. ActionExecutor JSON参数调用、契约信封、唯一输出目录、0/1标签与输入不变。
2. 缺失图片、错误尺寸、灰度输入被拒绝，未提前加载模型。
3. 模型加载失败和保存失败正确返回 API_ERROR。
4. 使用脚本化LLM响应验证ReAct执行工具并把真实工具返回送入下一轮提示词。
5. RGB float32 0..255 预处理符合固定上游版本。
6. 临时随机网络的旧格式全模型转换、严格state_dict加载与GPU前向、局部类路径映射、懒加载复用。

目录对齐搭档的三个算法后，模型位于 `models/Models.py`；六项测试再次全部通过。
独立推理、旧权重转换、Agent示例三种脚本的直接 `--help` 调用均通过，
其中直接脚本还检查了从工作目录之外启动；`-m algorithms.snunet.inference --help`
也通过。目录调整未改变模型算法。

已迁移到原项目根目录 `D:\AIProjects\Change-Agent`，当前分支 `feat/snunet-tool`。
代码与本地 `snunet-32.pt` 按哈希核对后迁移，原独立工作树已移除；
迁移后从根目录重新执行六项测试，全部通过。README 已改用根目录样例路径。

## 发布前补充验证 — 2026-11-18

真实官方 CDD 文件 `snunet-32.pt` 的本机 SHA256：
`05af4604fe73cd94c06a5758039216969df963a5e112cc0c9cfda3c3b428b710`。
该哈希用于固定本次输入文件，不是官方提供的校验值。

旧权重比固定源码多出 `ca2/ca3/ca4` 的六个张量。解析旧文件内嵌源码核对：
原始 `forward` 与当前版本 AST 一致，均只访问 `ca/ca1`。
转换脚本将六个未使用张量原样保存在 `unused_legacy_state`，其余参数严格加载。
已加入旧模块名、旧序列化格式、未使用参数保留和未知参数拒绝的回归测试，
现在七项测试全部通过。

以下为早期接入记录，后续发现遗漏池化配置，因此原数字不再作为原始CDD模型的效果证据：
真实权重成功转换，并在本机 RGB 256×256 演示图片上完成独立推理与
ActionExecutor 工具调用。A 对 A、A 对 B、B 对 A 的变化像素数分别为
0、32、323。A 对 B 的独立推理与工具输出一致，三次工具状态均为 SUCCESS。
原始权重、转换权重、输入图片与运行输出均未提交到 Git。

边界：样例是本机 LEVIR-MCI 演示图片，CDD 权重属于跨域使用；像素数不同
只能证明输入会影响输出，不证明精度或择优能力。没有计算 IoU/F1。
脚本化LLM测试只验证反馈链路；真实 DeepSeek 调用、同进程与搭档三个模型的
联合运行、完整精度/成本评估尚待验收。

## 完整测试前发现的转换一致性修正 — 2026-11-18

进一步解析旧权重内嵌的 ChannelAttention 类，发现其 max_pool 实际保存为
AdaptiveAvgPool2d，而当前固定源码构造 AdaptiveMaxPool2d。池化没有权重，
旧版严格 state_dict 检查无法发现该行为差异。已在转换产物中保存模型配置，
推理时恢复原池化；未修改上游模型源文件或图像预处理。

真实旧完整模型与修正后转换模型在演示A/B的 logits 最大绝对差为0，
输出完全一致；该样例变化像素为11694。旧版转换的32像素不代表原模型。
七项测试再次通过，并增加了转换前后前向输出一致性断言。
完整数据域评估结果另存本地 outputs/cdd-building-review-20261118，
本次修正和评估尚未自动提交或推送到GitHub。

修正后的完整 LEVIR-MCI 测试（1929对，建筑变化前景）结果：
SNUNet CDD IoU 8.7313%、F1 16.0603%、Precision 12.0452%、Recall 24.0906%；
MCI 建筑类别 IoU 81.9999%、F1 90.1098%、Precision 90.6531%、Recall 89.5730%。
固定并集 F1 49.1920%，固定交集 F1 34.5939%，均低于 MCI 单独使用。
878对建筑正样本中，SNUNet逐图IoU高于MCI的有47对；这仅为事后比较，
没有得到可部署的选择策略。因此不建议默认启用该CDD权重做建筑复核，
不将此结论外推为SNUNet架构本身无效。
