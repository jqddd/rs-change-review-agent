from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path
import sys
from threading import Lock
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from PIL import Image
from pydantic import BaseModel
import torch

from runtime import ROOT, load_model

mcp = FastMCP("mci")
model = lru_cache(maxsize=1)(load_model)
inference_lock = Lock()


class ChangeMask(BaseModel):
    mask_path: str
    road_pixels: int
    building_pixels: int


class ChangeCaption(BaseModel):
    caption: str


def validate_pair(a: str, b: str) -> None:
    for value in (a, b):
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"Provide an existing absolute local image path: {value}")
        with Image.open(path) as image:
            if (image.format not in {"PNG", "JPEG"} or image.mode != "RGB"
                    or image.size != (256, 256) or "transparency" in image.info):
                raise ValueError("Expected a fully valid 256x256 RGB PNG/JPEG tile")


@mcp.tool()
def mci_detect_changes(image_before: str, image_after: str) -> ChangeMask:
    """Segment road and building changes in aligned before/after RGB tiles. Labels: 0 background, 1 road, 2 building."""
    validate_pair(image_before, image_after)
    folder = ROOT / "test_out" / "mci_mcp" / uuid4().hex
    folder.mkdir(parents=True)
    with inference_lock, redirect_stdout(sys.stderr), torch.inference_mode():
        labels = model().change_detection(image_before, image_after, str(folder / "preview.png"))
    path = folder / "mask_label.png"
    Image.fromarray(labels).save(path)
    return ChangeMask(mask_path=str(path), road_pixels=int((labels == 1).sum()),
                      building_pixels=int((labels == 2).sum()))


@mcp.tool()
def mci_describe_changes(image_before: str, image_after: str) -> ChangeCaption:
    """Generate an English change caption directly from two aligned RGB 256x256 tiles."""
    validate_pair(image_before, image_after)
    with inference_lock, redirect_stdout(sys.stderr), torch.inference_mode():
        caption = model().generate_change_caption(image_before, image_after)
    return ChangeCaption(caption=caption.strip())


if __name__ == "__main__":
    mcp.run(transport="stdio")
