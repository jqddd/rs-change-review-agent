from getpass import getpass
import requests

key = getpass("请输入 DeepSeek API Key（输入不显示）：").strip()

response = requests.post(
    "https://api.deepseek.com/chat/completions",
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "deepseek-flash",
        "messages": [
            {"role": "user", "content": "请只回复：连接成功"}
        ],
        "thinking": {"type": "disabled"},
        "max_tokens": 64,
        "stream": False,
    },
    timeout=120,
)

print("HTTP 状态码：", response.status_code)

if response.ok:
    print("模型回复：", response.json()["choices"][0]["message"]["content"])
else:
    print("错误信息：", response.text)