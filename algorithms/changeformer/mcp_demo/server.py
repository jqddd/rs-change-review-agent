from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from PIL import Image
from pydantic import BaseModel, Field

from model_loader import ALGORITHM, load_model

mcp = FastMCP("changeformer")


class DetectionResult(BaseModel):
    mask_path: str = Field(description="Absolute server-local path to the 0/1 label PNG")
    changed_pixels: int = Field(ge=0, le=256 * 256, description="Number of building-change pixels")


@lru_cache(maxsize=1)
def model():
    return load_model()


@mcp.tool()
def changeformer_detect_building_change(image_before: str, image_after: str) -> DetectionResult:
    """Detect building changes in two aligned RGB 256x256 PNG/JPEG tiles.
    Pass absolute local paths in chronological order. Returns a label PNG path
    (0 unchanged, 1 building change) and the changed pixel count.
    """
    for path in (image_before, image_after):
        if not Path(path).is_absolute():
            raise ValueError("Use absolute local paths")
        if not Path(path).is_file():
            raise ValueError(f"Image file not found: {path}. Check the path on the server computer.")
        with Image.open(path) as image:
            image.load()
            if image.format not in ("PNG", "JPEG") or image.mode != "RGB" or image.size != (256, 256):
                raise ValueError("Expected RGB 256x256 PNG/JPEG")
            if any("nodata" in key.lower().replace("_", "") for key in image.info) or "transparency" in image.info:
                raise ValueError("Expected a fully valid tile")
    mask = model().detect(image_before, image_after)
    output = ALGORITHM.parents[1] / "test_out/changeformer_mcp" / uuid4().hex
    output.mkdir(parents=True)
    path = output / "mask_label.png"
    Image.fromarray(mask).save(path)
    return DetectionResult(mask_path=str(path), changed_pixels=int(mask.sum()))


if __name__ == "__main__":
    mcp.run(transport="stdio")
