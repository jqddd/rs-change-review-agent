# 在 Windows 上复现 TinyCD MCP

这是一项本地 CPU 推理服务。客户端通过 stdio 启动服务端，传入两期影像的路径，得到建筑变化标签 PNG 的路径和变化像素数。队友只需取得本示例、TinyCD 实现、权重和一对图片。Anthropic 的 mcp-builder 用于开发复核，运行示例只需要下面的 Python 环境。

## 1. 准备代码、权重和图片

在团队仓库运行 `git fetch origin feat/tinycd-mcp-stdio` 和 `git switch --detach FETCH_HEAD` 即可查看 PR 版本（建议使用干净的工作副本）。从仓库根目录打开 PowerShell。确认有 `algorithms/tinycd/inference.py`、同目录内的 `models/`、本目录中的两个 Python 脚本、`pyproject.toml` 和 `uv.lock`。权重放在 `algorithms/tinycd/pretrained_models/levir_best.pth`。可以从 [TinyCD 作者仓库](https://github.com/AndreaCodegoni/Tiny_model_4_CD/tree/main/pretrained_models) 获取。此次复核使用的权重 SHA256 为 `743645B87199C96BD0C3271AE5DF6F0918B3F1D84618DFB9396843CA08EE4A3E`。

```powershell
Get-FileHash .\algorithms\tinycd\pretrained_models\levir_best.pth
cd .\algorithms\tinycd\mcp_demo
```

## 2. 用锁文件准备独立环境

需要 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和 Python 3.12。先执行 `uv --version`。若 Windows 用户安装目录尚未加入 PATH，可在当前 PowerShell 中执行 `$env:Path += ";$env:USERPROFILE\.local\bin"`。首次安装依赖需要网络或已有 uv 缓存。

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe --version
```

`--locked` 使 uv 按提交的依赖版本创建环境。后面指定这个环境的 Python，客户端也会用同一个解释器启动服务端。

## 3. 先发现工具

```powershell
& .\.venv\Scripts\python.exe .\client.py
```

终端应显示 `detect_building_change required: ['image_before', 'image_after']`。此时客户端只读取工具清单。

## 4. 用 test_77 调用并验证

测试图来自团队 feat/bit-tool 分支。下面将这一对样例下载到临时目录，然后交给 TinyCD。使用自己的同规格影像时，直接设置 $a、$b 即可。

```powershell
$samples = Join-Path $env:TEMP ([guid]::NewGuid().ToString())
New-Item -ItemType Directory $samples | Out-Null
$a = Join-Path $samples A.png
$b = Join-Path $samples B.png
git fetch origin feat/bit-tool
& .\.venv\Scripts\python.exe -c "import pathlib, subprocess, sys; pathlib.Path(sys.argv[1]).write_bytes(subprocess.check_output(['git','show','FETCH_HEAD:algorithms/bit/samples/A/test_77_0512_0256.png']))" $a
& .\.venv\Scripts\python.exe -c "import pathlib, subprocess, sys; pathlib.Path(sys.argv[1]).write_bytes(subprocess.check_output(['git','show','FETCH_HEAD:algorithms/bit/samples/B/test_77_0512_0256.png']))" $b
Test-Path -LiteralPath $a
Test-Path -LiteralPath $b
& .\.venv\Scripts\python.exe .\client.py $a $b
```

前两个检查应返回 True。A 是较早影像，B 是较晚影像。两张图应为已经对齐、整块有效的 RGB 256×256 PNG/JPEG。客户端会将路径转成服务端可访问的绝对路径。服务端检查格式、尺寸和部分无效区标记，配准与像素有效性由调用方确认。

该样例的参考输出是 `changed_pixels: 11916` 和 `Direct Python and MCP masks match pixel by pixel.`。前者是模型预测的像素数，后者表示客户端又直接运行了一次 TinyCD，验证两张 Mask 完全一致。更换输入后，像素数会变化。评估精度需要另与标注比较。

## 5. 打开本次生成的 Mask

输出在仓库的 `test_out/tinycd_mcp/<本次调用ID>/mask_label.png`。把下面的 `实际ID` 替换为终端返回的目录名。

```powershell
$mask = '..\..\..\test_out\tinycd_mcp\实际ID\mask_label.png'
& .\.venv\Scripts\python.exe -c "from PIL import Image; import sys; Image.open(sys.argv[1]).point(lambda p: 255 if p == 1 else 0).show()" $mask
```

标签图用 0 表示背景、1 表示建筑变化，直接打开时几乎全黑。这条命令将 1 显示为白色，保留原始标签文件。

## 复核记录

本示例完成后使用 [Anthropic mcp-builder](https://github.com/anthropics/skills/tree/main/skills/mcp-builder) 复核。接口名称和说明给出了用途、输入顺序、尺寸与输出编码。每次成功调用写入独立目录。故意传入不存在的图片时，工具返回文件路径错误，客户端输出工具错误并以状态码 1 退出。可先用上面的 Test-Path 定位。torchvision 的 pretrained 弃用提示可与正常成功结果同时出现。

客户端发现工具时还会显示 Result fields。输出 schema 将 mask_path 声明为字符串、changed_pixels 声明为整数，客户端直接读取结构化结果。本轮验证范围是本地 CPU、stdio 协议调用和封装一致性，Agent 自主选工具与 GPU 推理需要分别验证。
