"""DDPM-CD-only ReAct demo; no general Python interpreter is enabled."""
from __future__ import annotations

import argparse
from getpass import getpass
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lagent.actions import ActionExecutor
from lagent.agents import ReAct
from lagent.llms import GPTAPI
from lagent.schema import ActionStatusCode
from tools.ddpm_cd_tool import DDPMCDTool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-weight")
    parser.add_argument("--change-weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    tool = DDPMCDTool(args.diffusion_weight, args.change_weight, args.device, args.seed)
    tool._get_model()  # Fail on missing/incompatible weights before a paid API call.
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
        raise SystemExit("UNVERIFIED: no successful DDPM-CD execution; text alone is not a detection result")
    print("VERIFIED TOOL EXECUTION\n", result.response)


if __name__ == "__main__":
    main()
