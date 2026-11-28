"""Run requested measurements on the accepted label mask for each scene."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from jinja2 import Environment, StrictUndefined


if __package__:
    from .spatial import spatial_metrics
    from .building_pixels import pixel_counts, read_label_png
else:
    from spatial import spatial_metrics
    from building_pixels import pixel_counts, read_label_png


# Values describe the upstream label encoding, including background.
SCHEMES = {"tinycd": {"background": 0, "building": 1},
           "mci": {"background": 0, "road": 1, "building": 2}}
METRICS = {"pixels", "regions", "area_m2", "description", "roi", "comparison", "vectors", "zones"}


def read_grid(path):
    if path.suffix.lower() not in {".tif", ".tiff"}:
        labels = read_label_png(path)
        return labels, np.ones(labels.shape, dtype=bool), None
    import rasterio

    with rasterio.open(path) as src:
        if src.count != 1 or not np.issubdtype(src.dtypes[0], np.integer):
            raise ValueError("Expected one integer label band")
        area = None
        if src.crs and src.crs.is_projected and not src.transform.is_identity:
            t = src.transform
            area = abs(t.a * t.e - t.b * t.d) * src.crs.linear_units_factor[1] ** 2
        return src.read(1), src.read_masks(1) != 0, area


def describe(source, root, shape, fully_valid):
    if "mci" not in source:
        return {"status": "missing_input", "detail": "需要同一图块的两期原图和 MCI 环境"}
    if shape != (256, 256) or not fully_valid:
        return {"status": "unsupported_input", "detail": "MCI 描述要求全有效的 256×256 图块"}
    cfg = source["mci"]
    runtime = Path(os.environ.get("MCI_RUNTIME_DIR", Path(__file__).resolve().parents[3] / "mci-mcp/Multi_change/mcp_demo"))
    python = Path(os.environ.get("MCI_PYTHON", runtime / ".venv/Scripts/python.exe"))
    args = [python, Path(__file__).with_name("mci_caption.py"),
            runtime, root / cfg["before"], root / cfg["after"]]
    run = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace")
    if run.returncode:
        return {"status": "error", "detail": run.stderr.strip()[-1200:]}
    return {"status": "ok", "model": "MCI", "scope": "whole_image_pair",
            "before": cfg["before"], "after": cfg["after"], "text": json.loads(run.stdout)}


def measure(scene, wanted, source, root):
    encoding = SCHEMES.get(source["scheme"], {})
    if scene == "background" or scene not in encoding:
        return {"status": "unsupported_scheme", "scheme": source["scheme"]}
    unknown = set(wanted) - METRICS
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    path = root / source["mask"]
    values, valid, grid_area = read_grid(path)
    if "valid_mask" in source:
        coverage = read_label_png(root / source["valid_mask"])
        if coverage.shape != values.shape or not np.isin(coverage, [0, 1]).all():
            raise ValueError("Expected a same-grid 0/1 valid mask")
        valid &= coverage == 1
    elif path.suffix.lower() == ".png" and source.get("fully_valid") is not True:
        raise ValueError("PNG requires fully_valid=true or a valid_mask")
    if not np.isin(values[valid], list(encoding.values())).all():
        raise ValueError("Unknown label inside valid coverage")
    changed = valid & (values == encoding[scene])
    count, total, share = pixel_counts(changed, int(valid.sum()))
    result = {"status": "ok", "mask": source["mask"], "scheme": source["scheme"]}
    if "pixels" in wanted:
        result["pixels"] = dict(changed=count, valid=total, share=share)
    for metric in ("roi", "zones"):
        if metric in wanted:
            try:
                spatial_metrics([metric], source, root, path, changed, valid, result)
            except (OSError, KeyError, ValueError, TypeError) as exc:
                result[metric] = {"status": "error", "detail": str(exc)}
    region_metrics = set(wanted) & {"regions", "comparison", "vectors"}
    if region_metrics:
        spatial_metrics(region_metrics, source, root, path, changed, valid, result)
    if "area_m2" in wanted:
        area = source.get("pixel_area_m2", grid_area)
        if area is None:
            result["area_m2"] = {"status": "missing_scale"}
        elif not math.isfinite(area) or area <= 0:
            raise ValueError("Pixel area must be finite and positive")
        else:
            result["area_m2"] = dict(status="ok", value=count * area, pixel_area=area,
                                     source="provided" if "pixel_area_m2" in source else "projected_grid")
    if "description" in wanted:
        try:
            result["description"] = describe(source, root, values.shape, valid.all())
        except (KeyError, ValueError, OSError) as exc:
            result["description"] = {"status": "error", "detail": str(exc)}
    return result


def analyze(request, root):
    results = {}
    for scene, wanted in request["questions"].items():
        source = request["inputs"].get(scene)
        if source is None:
            results[scene] = {"status": "missing_mask"}
            continue
        try:
            results[scene] = measure(scene, wanted, source, root)
        except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
            results[scene] = {"status": "error", "detail": str(exc)}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = analyze(json.loads(args.request.read_text(encoding="utf-8-sig")), args.request.resolve().parent)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        template = Path(__file__).with_name("report.md.j2").read_text(encoding="utf-8")
        print(Environment(undefined=StrictUndefined).from_string(template).render(results=results))
