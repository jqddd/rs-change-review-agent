# BIT 建筑变化检测 MCP

在本目录打开 PowerShell。准备 Python 3.12、uv，按 [权重说明](../WEIGHTS.md) 将 LEVIR-CD 模型放到 `../checkpoints/best_ckpt.pt`。样例已随 BIT 分支提供。

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe client.py
& .\.venv\Scripts\python.exe client.py ../samples/A/test_77_0512_0256.png ../samples/B/test_77_0512_0256.png
```

第一条客户端命令通过 stdio 发现 `bit_detect_building_change`。第二条将变化前后两张已配准、全幅有效的 RGB 256×256 图像交给服务端，返回单通道 0/1 PNG 路径和预测建筑变化像素数。客户端随后用同一权重做一次直接推理，并逐像素核对。此示例使用 CPU。Agent 可使用相同的环境解释器以 `server.py` 为参数启动服务；传入的绝对路径必须在服务端可读。
