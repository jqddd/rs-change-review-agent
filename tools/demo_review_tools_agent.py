"""Registered finite-tool demo: CPU smoke, or explicit DeepSeek ReAct calls."""
from __future__ import annotations

import argparse
from getpass import getpass
import json
import os
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from lagent.actions import ActionExecutor
from lagent.agents import ReAct
from lagent.schema import ActionReturn, ActionStatusCode
from tools.image_quality_tool import ImageQualityCheckTool
from tools.alignment_check_tool import AlignmentCheckTool
from tools.registration_tool import RegistrationCorrectionTool


class ReviewActionExecutor(ActionExecutor):
    """Local demo budget, not a new shared agent framework or adaptive policy."""
    def __init__(self, max_calls=6, max_registration_calls=2):
        if (type(max_calls) is not int or type(max_registration_calls) is not int
                or not 1<=max_calls<=20 or not 1<=max_registration_calls<=max_calls):
            raise ValueError('Invalid fixed review-tool budgets')
        super().__init__([ImageQualityCheckTool(),AlignmentCheckTool(),RegistrationCorrectionTool()])
        self.max_calls,self.max_registration_calls = max_calls,max_registration_calls
        self.calls,self.registration_calls = 0,0

    def __call__(self,name,command):
        # These are standalone actions. The shared executor's dotted API aliases
        # must not bypass budgets or reach undeclared/private methods.
        if not isinstance(name,str) or '.' in name:
            return ActionReturn(type=str(name),state=ActionStatusCode.ARGS_ERROR,
                errmsg='Use the declared standalone tool name without a dotted API alias')
        if name in self.actions:
            if self.calls>=self.max_calls or (name=='registration_correct' and self.registration_calls>=self.max_registration_calls):
                return ActionReturn(type=name,state=ActionStatusCode.API_ERROR,errmsg='Fixed tool-call budget exhausted; stop or request human review')
            self.calls+=1
            if name=='registration_correct':self.registration_calls+=1
        return super().__call__(name,command)


def make_executor():
    return ReviewActionExecutor()


def smoke(image_before,image_after,output_dir):
    """Explicit executor invocation; not LLM planning or autonomous selection."""
    executor=make_executor()
    command=json.dumps(dict(image_before=image_before,image_after=image_after,output_dir=output_dir),ensure_ascii=False)
    trace=[]
    for name in executor.action_names():
        action=executor(name,command)
        trace.append(dict(tool=action.type,state=action.state.name,errmsg=action.errmsg,result=action.result))
        print(json.dumps(trace[-1],ensure_ascii=False))
        if action.state!=ActionStatusCode.SUCCESS:
            raise RuntimeError('Tool invocation failed: '+str(action.errmsg))
    return trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-before')
    parser.add_argument('--image-after')
    parser.add_argument('--output-dir',default='output/review_tools')
    parser.add_argument('--smoke',action='store_true',help='Explicit CPU executor calls; no API request')
    parser.add_argument('--question',help='Natural-language request for the registered ReAct tools')
    parser.add_argument('--model',default='deepseek-flash',help='Use a model available to your existing DeepSeek account')
    parser.add_argument('--endpoint',default='https://api.deepseek.com/chat/completions')
    args=parser.parse_args()
    if args.smoke:
        if not args.image_before or not args.image_after:parser.error('--smoke needs both image paths')
        trace=smoke(args.image_before,args.image_after,args.output_dir)
        print('VERIFIED EXECUTOR CALLS:',','.join(item['tool'] for item in trace))
        return
    if not args.question:parser.error('Choose --smoke or supply --question')
    from lagent.llms import GPTAPI
    key=os.environ.get('DEEPSEEK_API_KEY') or getpass('DeepSeek API Key (hidden): ')
    llm=GPTAPI(model_type=args.model,key=key,openai_api_base=args.endpoint,temperature=.1,max_new_tokens=3072)
    llm.gen_params['thinking']={'type':'disabled'}
    executor=make_executor()
    agent=ReAct(llm=llm,action_executor=executor,max_turn=8)
    result=agent.chat(args.question)
    executed=[]
    for action in result.actions:
        print('TOOL:',action.type,'STATE:',action.state,'RESULT:',action.result,'ERROR:',action.errmsg)
        if action.type in executor.action_names() and action.state==ActionStatusCode.SUCCESS:executed.append(action.type)
    if not executed:raise SystemExit('UNVERIFIED: no successful review-tool call')
    print('VERIFIED TOOL EXECUTION:',','.join(executed))
    print(result.response)


if __name__=='__main__':main()
