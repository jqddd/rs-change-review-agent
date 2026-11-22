"""SNUNet-only ReAct demo. API key is entered locally, never saved."""
from __future__ import annotations

import argparse
from getpass import getpass
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lagent.actions import ActionExecutor
from lagent.agents import ReAct
from lagent.llms import GPTAPI
from lagent.schema import ActionStatusCode

from tools.snunet_tool import SNUNetTool
from algorithms.snunet.inference import DEFAULT_PROFILE, PROFILE_SETTINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", help="Defaults to the selected profile's converted checkpoint")
    parser.add_argument("--profile", choices=tuple(PROFILE_SETTINGS), default=DEFAULT_PROFILE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trained-on", help="Optional check; must match the fixed deployment profile")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
    args = parser.parse_args()
    # A missing checkpoint fails before making any paid API call.
    tool = SNUNetTool(args.weight, args.device, args.trained_on, args.profile)
    tool._get_model()
    key = os.environ.get("DEEPSEEK_API_KEY") or getpass("DeepSeek API Key (hidden): ")
    llm = GPTAPI(model_type=args.model, key=key, openai_api_base=args.endpoint,
                 temperature=0.1, max_new_tokens=2048)
    llm.gen_params["thinking"] = {"type": "disabled"}
    agent = ReAct(llm=llm, action_executor=ActionExecutor(actions=[tool]))
    result = agent.chat([{"role": "user", "content": args.question}])
    executed = False
    for action in result.actions:
        print("TOOL:", action.type, "STATE:", action.state, "RESULT:", action.result, "ERROR:", action.errmsg)
        executed |= action.type == tool.name and action.state == ActionStatusCode.SUCCESS
    if not executed:
        raise SystemExit("UNVERIFIED: no successful SNUNet tool execution; model text is not a detection result")
    print("VERIFIED TOOL EXECUTION\n", result.response)


if __name__ == "__main__":
    main()
