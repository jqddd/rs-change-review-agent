"""ChangeFormer 工具的 Agent 演示（验证契约第 9 节第 4 条：Agent 可调用该工具）。

用法：
    set DEEPSEEK_API_KEY=你的key
    python tools/demo_changeformer_agent.py --question "检测 ... 的建筑变化"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (str(ROOT / "tools"), str(ROOT / "algorithms" / "changeformer")):
    if d not in sys.path:
        sys.path.insert(0, d)

from lagent.actions import ActionExecutor  # noqa: E402
from lagent.agents import ReAct  # noqa: E402
from lagent.llms import GPTAPI  # noqa: E402

from changeformer_tool import ChangeFormerTool  # noqa: E402


def build_agent() -> ReAct:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    llm = GPTAPI(model_type=model_name, key=api_key, openai_api_base=base_url)
    tool = ChangeFormerTool()
    return ReAct(llm=llm, action_executor=ActionExecutor(actions=[tool]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, help="单次提问")
    args = parser.parse_args()

    chatbot = build_agent()
    question = args.question or input("请输入问题：")

    history = [dict(role="user", content=question)]
    print("||Start thinking ...")
    result = chatbot.chat(history)

    print("\n=== 工具调用记录 ===")
    for action in result.actions:
        print(f"工具: {action.type} | 状态: {action.state}")
        if action.args:
            print(f"  参数: {action.args}")
        if action.result:
            print(f"  结果: {action.result}")
        if action.errmsg:
            print(f"  错误: {action.errmsg}")

    print("\n=== Agent 回答 ===")
    print(result.response)


if __name__ == "__main__":
    main()
