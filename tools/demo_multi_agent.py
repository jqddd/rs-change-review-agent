"""多算法择优演示：把 5 个变化检测工具注册进一个 Agent，让 LLM 选择。

用法：
    单次：python tools/demo_multi_agent.py --question "..."
    交互：python tools/demo_multi_agent.py   （输入 exit 退出）

交互时可以直接把图片路径粘进去，例如：
    检测 D:/.../sample_02_A.png 和 D:/.../sample_02_B.png 之间的建筑变化
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (str(ROOT), str(ROOT / "tools")):
    if d not in sys.path:
        sys.path.insert(0, d)

from lagent.actions import ActionExecutor  # noqa: E402
from lagent.agents import ReAct  # noqa: E402
from lagent.llms import GPTAPI  # noqa: E402

from bit_tool import BITTool  # noqa: E402
from changeformer_tool import ChangeFormerTool  # noqa: E402
from ddpm_cd_tool import DDPMCDTool  # noqa: E402
from snunet_tool import SNUNetTool  # noqa: E402
from tinycd_tool import TinyCDTool  # noqa: E402
from tools.image_quality_tool import ImageQualityCheckTool
from tools.preprocess_tool import DenoiseTool, HistogramMatchTool, DeblurTool

TOOL_NAMES = {"changeformer", "bit", "tinycd", "snunet", "ddpm_cd"}


def make_executor() -> ActionExecutor:
    return ActionExecutor(actions=[
        ChangeFormerTool(), BITTool(), TinyCDTool(), SNUNetTool(), DDPMCDTool(),
        ImageQualityCheckTool(), DenoiseTool(), HistogramMatchTool(), DeblurTool(),
    ])


def build_agent() -> ReAct:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请设置环境变量 DEEPSEEK_API_KEY")
    llm = GPTAPI(
        model_type=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        key=api_key,
        openai_api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"),
    )
    return ReAct(llm=llm, action_executor=make_executor())


def run_turn(chatbot: ReAct, history: list) -> str:
    """跑一轮，打印工具调用情况，返回最终回答。"""
    result = chatbot.chat(history)
    for a in result.actions:
        print(f"    [工具调用] {a.type} -> {a.state}")
    if not any(a.type in TOOL_NAMES for a in result.actions):
        print("    [警告] 未调用任何变化检测工具（可能是幻觉）")
    return result.response


def main() -> None:
    parser = argparse.ArgumentParser(description="多算法择优演示")
    parser.add_argument("--question", default=None, help="单次提问（不提供则进入交互模式）")
    args = parser.parse_args()

    chatbot = build_agent()

    if args.question:
        history = [dict(role="user", content=args.question)]
        print(run_turn(chatbot, history))
        return

    # 交互模式
    print("=" * 50)
    print("多算法择优 Agent 已启动（5 个变化检测算法可选）")
    print("输入 exit 退出。可以直接粘图片路径提问。")
    print("=" * 50)
    history = []
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue
        history.append(dict(role="user", content=user_input))
        print("Agent 思考中...")
        response = run_turn(chatbot, history)
        history.append(dict(role="assistant", content=response))
        print("\nAgent:", response)


if __name__ == "__main__":
    main()
