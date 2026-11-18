# 验证记录 — 2026-11-17

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

真实权重成功转换，并在本机 RGB 256×256 演示图片上完成独立推理与
ActionExecutor 工具调用。A 对 A、A 对 B、B 对 A 的变化像素数分别为
0、32、323。A 对 B 的独立推理与工具输出一致，三次工具状态均为 SUCCESS。
原始权重、转换权重、输入图片与运行输出均未提交到 Git。

边界：样例是本机 LEVIR-MCI 演示图片，CDD 权重属于跨域使用；像素数不同
只能证明输入会影响输出，不证明精度或择优能力。没有计算 IoU/F1。
脚本化LLM测试只验证反馈链路；真实 DeepSeek 调用、同进程与搭档三个模型的
联合运行、完整精度/成本评估尚待验收。
