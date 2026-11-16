import os
from getpass import getpass

from lagent.agents import ReAct
from lagent.actions import (
    ActionExecutor,
    Visual_Change_Process_PythonInterpreter,
)
from lagent.llms import GPTAPI

# 保证模型权重和词表的相对路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

key = getpass("请输入 DeepSeek API Key（输入不显示）：").strip()

llm = GPTAPI(
    model_type="deepseek-flash",
    key=key,
    openai_api_base="https://api.deepseek.com/chat/completions",
    temperature=0.1,
    max_new_tokens=2048,
)
llm.gen_params["thinking"] = {"type": "disabled"}

chatbot = ReAct(
    llm=llm,
    action_executor=ActionExecutor(
        actions=[Visual_Change_Process_PythonInterpreter()]
    ),
)

history = []
print("Change-Agent 已启动，输入 exit 退出。")

while True:
    question = input("\n你：").strip()
    if question.lower() in {"exit", "quit"}:
        break
    if not question:
        continue

    history.append({"role": "user", "content": question})
    print("正在处理，请稍候……")

    result = chatbot.chat(history)

    # 显示工具执行记录，确认实际调用了模型
    for action in result.actions:
        print("\n工具：", action.type)
        print("状态：", action.state)
        if action.errmsg:
            print("错误：", action.errmsg)
        if action.result:
            print("工具结果：", action.result)

    print("\nChange-Agent：", result.response)
    history.append({
        "role": "assistant",
        "content": result.response,
    })