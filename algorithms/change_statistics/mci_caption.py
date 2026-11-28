"""Call the existing CPU MCI Python runtime in its own environment."""
from contextlib import redirect_stdout
import json
import sys

from PIL import Image
import torch


if __name__ == "__main__":
    runtime_dir, before, after = sys.argv[1:]
    for path in (before, after):
        with Image.open(path) as image:
            if image.size != (256, 256) or image.mode != "RGB" or "transparency" in image.info:
                raise ValueError("Expected an aligned, fully valid 256x256 RGB pair")
    sys.path.insert(0, runtime_dir)
    with redirect_stdout(sys.stderr), torch.inference_mode():
        from runtime import load_model
        caption = load_model().generate_change_caption(before, after).strip()
    if not caption:
        raise ValueError("MCI returned an empty caption")
    print(json.dumps(caption))
