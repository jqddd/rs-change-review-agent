# MCI 本地 MCP 服务

在本目录打开 Windows PowerShell，准备 Python 3.12 和 uv。从 [Change-Agent 作者仓库](https://github.com/Chen-Yang-Liu/Change-Agent) 获取 MCI_model.pth，存放于 ../models_ckpt/。服务还使用分支中的 MiT-B1 骨干权重、词表与 scripts/stubs。

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe client.py
```

客户端通过 stdio 启动同一环境中的 server.py。无参数时显示工具及返回字段。mci_detect_changes 返回三分类标签 PNG（0 背景、1 道路、2 建筑）与分类型像素数，mci_describe_changes 返回英文描述。

## 复现 test_77

以下命令从团队 BIT 分支提取两张公开样例图到临时目录。使用自己的数据时，将 $a 和 $b 换成自己电脑上的影像路径即可。两图按时间先后排列，须已配准，整块有效，格式为 256×256 RGB PNG/JPEG。

```powershell
git fetch origin feat/bit-tool
$sample = New-Item -ItemType Directory (Join-Path $env:TEMP ([guid]::NewGuid().ToString()))
foreach ($epoch in 'A','B') {
    & .\.venv\Scripts\python.exe -c "import pathlib, subprocess, sys; pathlib.Path(sys.argv[1]).write_bytes(subprocess.check_output(['git','show',sys.argv[2]]))" (Join-Path $sample.FullName "$epoch.png") "FETCH_HEAD:algorithms/bit/samples/$epoch/test_77_0512_0256.png"
}
$a = Join-Path $sample.FullName A.png
$b = Join-Path $sample.FullName B.png
& .\.venv\Scripts\python.exe client.py $a $b
```

客户端先检查 MCP 调用成功，再调用同一个算法的 Python 推理入口，逐像素核对 Mask，并核对描述文本。输出位于仓库 test_out/mci_mcp/ 下的本次调用目录。低值标签图直接打开时接近黑色。该对照验证 CPU 上封装的一致性，评估精度需要与人工标签比较。

接入其他 MCP 客户端时，启动命令填写本目录 .venv/Scripts/python.exe 的绝对路径，参数填写 server.py 的绝对路径，传输方式选 stdio。客户端传入的图片路径须在服务端可读。
