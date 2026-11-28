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
            result = await session.call_tool("bit_detect_building_change", {"image_before": a, "image_after": b})
            if result.isError:
                for item in result.content:
                    if item.type == "text":
                        print(item.text, file=sys.stderr)
                return 1
            record = result.structuredContent
            print(record, flush=True)
            import numpy as np
            from PIL import Image
            from model_loader import load_model
            direct = load_model()
            with Image.open(record["mask_path"]) as image:
                np.testing.assert_array_equal(np.asarray(image), direct.detect(a, b))
            print("Direct Python and MCP masks match pixel by pixel.")


if __name__ == "__main__":
    if len(sys.argv) not in (1, 3):
        raise SystemExit("Usage: client.py [A.png B.png]")
    sys.exit(asyncio.run(main(sys.argv[1:])))
