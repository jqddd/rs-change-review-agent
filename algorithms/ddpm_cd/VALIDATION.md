# DDPM-CD 接入验证记录

日期：2026-11-18。开发分支 `feat/ddpm-cd-tool`，公共基线
`eb54e664ab55c7eae6c69e3422de8ceeb620e4ec`。

## 已完成

- 官方最小推理源码、MIT LICENSE、固定commit与逐文件来源记录已保存。
- 独立推理和Lagent工具分离，模型源码放在models目录；不改公共Agent核心。
- 明确两组权重、decoder特征、三时间步与官方预处理；不运行训练/日志服务。
- 现有change-agent环境可导入；未安装或升级依赖。
- `python -m unittest discover -s tests -p "test_ddpm_cd*.py" -v`：12项全部通过。
- 独立推理和 Agent 示例的直接脚本、`python -m` 四种 `--help` 入口均通过；直接脚本也在项目外工作目录验证。

测试分层：

1. 官方小型随机 UNet + GaussianDiffusion + 三步变化头真实前向，输出shape正确。
2. 全特征与仅保留指定槽位的特征，检测头logits逐位一致。
3. 固定噪声种子可重复，执行后恢复NumPy与Torch状态，包括异常路径。
4. 完整默认网络仅meta构造：扩散391,048,067参数，检测头46,405,126参数。
5. ActionExecutor JSON参数、独立目录、单通道0/1标签、可视化和运行记录。
6. 文件缺失、错误尺寸/通道、推理/写文件失败返回正确错误状态。
7. tensor权重严格加载、module前缀、ZIP mmap及旧格式fallback的兼容检查。
8. 双权重懒加载复用、官方同名文件的不完整大小检测。
9. 使用脚本化LLM验证ReAct调用和结果回传，不代表真实DeepSeek自主选择。

## 尚未验收

用户选择通过浏览器下载官方权重并提供路径。目前未加载完整官方权重，
因此真实样例预测、8GB显存峰值、延迟、模型精度、原官方推理结果一致性以及
真实DeepSeek工具调用均未验证。小型随机模型、假检测器和meta结构不能替代这些检查。

本分支仅包含DDPM-CD接入，不包含SNUNet算法提交。
权重、临时下载文件、推理产物和本机缓存不纳入提交。
