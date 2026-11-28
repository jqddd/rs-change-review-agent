import asyncio
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(paths):
    server = Path(__file__).with_name("server.py").resolve()
    runtime_env = {k: os.environ[k] for k in ("TEMP", "TMP", "LOCALAPPDATA", "WINDIR", "SystemRoot", "ProgramFiles", "ProgramFiles(x86)") if k in os.environ}
    params = StdioServerParameters(command=sys.executable, args=[str(server)], env=runtime_env)
    async with stdio_client(params) as streams, ClientSession(*streams) as session:
        await session.initialize()
        for tool in (await session.list_tools()).tools:
            print(tool.name, "required:", tool.inputSchema["required"], flush=True)
            print("Result fields:", list(tool.outputSchema["properties"]), flush=True)
        if paths:
            a, b = (str(Path(p).resolve()) for p in paths)
            results = {}
            for name in ("mci_detect_changes", "mci_describe_changes"):
                result = await session.call_tool(name, {"image_before": a, "image_after": b})
                if result.isError:
                    for item in result.content:
                        if item.type == "text":
                            print(item.text, file=sys.stderr)
                    return 1
                results[name] = result.structuredContent
                print(name, result.structuredContent, flush=True)
    if paths:
        from tempfile import TemporaryDirectory
        import numpy as np
        import torch
        from PIL import Image
        from runtime import load_model

        with TemporaryDirectory() as temp, torch.inference_mode():
            direct = load_model()
            labels = direct.change_detection(a, b, str(Path(temp) / "preview.png"))
            with Image.open(results["mci_detect_changes"]["mask_path"]) as image:
                np.testing.assert_array_equal(np.asarray(image), labels)
            caption = direct.generate_change_caption(a, b).strip()
            if caption != results["mci_describe_changes"]["caption"]:
                raise AssertionError("MCP caption differs from direct inference")
        print("Direct Python and MCP masks and captions match.")


if __name__ == "__main__":
    if len(sys.argv) not in (1, 3):
        raise SystemExit("Usage: client.py [A.png B.png]")
    sys.exit(asyncio.run(main(sys.argv[1:])))
