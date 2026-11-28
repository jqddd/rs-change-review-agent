"""Spatial measurements on a single accepted label grid."""
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries


def locate(changed):
    labels = label(changed, connectivity=2)
    props = sorted((p for p in regionprops(labels) if p.area >= 6),
                   key=lambda p: (-p.area, p.label))
    ids = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    rows = []
    for index, prop in enumerate(props, 1):
        ids[prop.label] = index
        y, x, bottom, right = prop.bbox
        rows.append(dict(id=index, area_pixels=int(prop.area), bbox=[x, y, right, bottom],
                         centroid_xy=list(reversed(prop.centroid))))
    return ids[labels], rows


def compare(grid, rows, cfg, root):
    number = cfg.get("region_id", 1)
    if type(number) is not int or not 1 <= number <= len(rows):
        raise ValueError("Unknown region_id")
    bounds = rows[number - 1]["bbox"]
    crops = []
    for key in ("before", "after"):
        with Image.open(root / cfg[key]) as image:
            if image.size != grid.shape[::-1]:
                raise ValueError("Comparison requires aligned same-grid images")
            crops.append(image.convert("RGB").crop(bounds))
    outline = find_boundaries(np.pad(grid == number, 1), mode="inner")[1:-1, 1:-1]
    marked = crops[1].copy()
    marked.paste((255, 60, 30), mask=Image.fromarray(outline).crop(bounds))
    width, height = marked.size
    canvas = Image.new("RGB", (3 * width, height))
    for i, panel in enumerate([*crops, marked]):
        canvas.paste(panel, (i * width, 0))
    output = root / f"region-{number}-{uuid4().hex}.png"
    canvas.save(output)
    return dict(region_id=number, path=str(output), panels=["before", "after", "after_outline"])


def polygons(grid, transform):
    for geometry, value in shapes(grid, mask=grid != 0, transform=transform):
        yield int(value), shape(geometry)


def grouped_geometry(grid, transform):
    parts = {}
    for number, geom in polygons(grid, transform):
        parts.setdefault(number, []).append(geom)
    return {k: unary_union(v) for k, v in parts.items()}


def spatial_metrics(wanted, source, root, path, changed, valid, result):
    transform = rasterio.Affine.identity()
    crs = None
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as ds:
            transform, crs = ds.transform, ds.crs
    if set(wanted) & {"regions", "comparison", "vectors"}:
        grid, rows = locate(changed)
        if "regions" in wanted:
            result["regions"] = dict(count=len(rows), sizes=[r["area_pixels"] for r in rows],
                                     connectivity=8, min_pixels=6, items=rows)
        if "comparison" in wanted:
            try:
                result["comparison"] = compare(grid, rows, source["comparison"], root)
            except (ValueError, OSError, KeyError) as exc:
                result["comparison"] = {"status": "error", "detail": str(exc)}
        if "vectors" in wanted:
            features = []
            for number, geom in grouped_geometry(grid, transform if crs else rasterio.Affine.identity()).items():
                geometry = mapping(geom)
                if crs:
                    geometry = transform_geom(crs, "EPSG:4326", geometry)
                features.append(dict(type="Feature", properties=rows[number-1], geometry=geometry))
            suffix = "geojson" if crs else "pixels.json"
            out = root / f"regions-{uuid4().hex}.{suffix}"
            out.write_text(json.dumps(dict(type="FeatureCollection", features=features)), encoding="utf-8")
            result["vectors"] = dict(path=str(out), coordinates="WGS84" if crs else "pixel_xy")
    if "roi" in wanted:
        from rasterio.features import geometry_mask
        cfg = source["roi"]
        geom = box(*cfg["bbox"]) if "bbox" in cfg else shape(cfg["geometry"])
        if geom.geom_type not in {"Polygon","MultiPolygon"} or not geom.is_valid or geom.is_empty:
            raise ValueError("Invalid ROI polygon")
        inside = valid & geometry_mask([mapping(geom)], valid.shape, rasterio.Affine.identity(), invert=True)
        count, total = int((changed & inside).sum()), int(inside.sum())
        result["roi"] = dict(changed=count, valid=total, share=count/total if total else None)
    if "zones" in wanted:
        if not crs or not crs.is_projected:
            raise ValueError("Zones require a projected GeoTIFF")
        boundaries = json.loads((root / source["zones"]).read_text(encoding="utf-8-sig"))
        changed_geom = unary_union([g for _, g in polygons(changed.astype("uint8"), transform)])
        valid_geom = unary_union([g for _, g in polygons(valid.astype("uint8"), transform)])
        result["zones"] = []
        for f in boundaries["features"]:
            geom = shape(transform_geom("EPSG:4326", crs, f["geometry"]))
            if geom.geom_type not in {"Polygon","MultiPolygon"} or not geom.is_valid:
                raise ValueError("Invalid zone polygon")
            scale = crs.linear_units_factor[1] ** 2
            area = geom.intersection(changed_geom).area * scale
            coverage = geom.intersection(valid_geom).area * scale
            result["zones"].append(dict(name=f["properties"]["name"], changed_m2=area,
                                        valid_m2=coverage, share=area/coverage if coverage else None))
