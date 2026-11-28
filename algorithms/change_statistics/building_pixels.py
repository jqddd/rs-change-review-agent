"""Count predicted building-change pixels in a fully valid TinyCD label PNG."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_label_png(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError("Expected a single-channel 8-bit label PNG")
        return np.asarray(image)


def pixel_counts(changed: np.ndarray, valid_count: int) -> tuple[int, int, float]:
    if valid_count == 0:
        raise ValueError("No valid pixels")
    count = int(np.count_nonzero(changed))
    return count, valid_count, count / valid_count


def count_building_pixels(mask_path: Path) -> tuple[int, int, float]:
    """Return changed pixels, total pixels and their fraction (0 to 1)."""
    labels = read_label_png(mask_path)

    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Expected TinyCD labels 0 and 1; use the original label mask")

    return pixel_counts(labels == 1, labels.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_path", type=Path)
    args = parser.parse_args()
    changed, total, fraction = count_building_pixels(args.mask_path)
    print(f"Changed pixels: {changed}")
    print(f"Total pixels:   {total}")
    print(f"Changed share:  {fraction:.2%}")
