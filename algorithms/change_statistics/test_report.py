import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping

from report import analyze


class ReportTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.values = np.zeros((6, 8), dtype="uint8")
        self.values[1:4, 2:6] = 1
        self.values[5, 7] = 1
        Image.fromarray(self.values).save(self.root / "mask.png")
        self.src = dict(mask="mask.png", scheme="tinycd", fully_valid=True)

    def run_metrics(self, *items):
        req = dict(questions={"building": items, "forest": ["pixels"]}, inputs={"building": self.src})
        out = analyze(req, self.root)
        self.assertEqual(out["forest"]["status"], "missing_mask")
        json.dumps(out)
        return out["building"]

    def test_spatial(self):
        self.src.update(roi={"bbox":[2, 1, 4, 4]}, comparison={"before":"mask.png","after":"mask.png"})
        r = self.run_metrics("pixels","regions","comparison","vectors","roi","area_m2")
        self.assertEqual(r["pixels"]["changed"], 13)
        self.assertEqual(r["regions"]["items"][0]["bbox"], [2,1,6,4])
        self.assertEqual(r["roi"], dict(changed=6, valid=6, share=1))
        self.assertEqual(r["area_m2"]["status"], "missing_scale")
        with Image.open(r["comparison"]["path"]) as img:
            self.assertEqual(img.size, (12,3))
            self.assertEqual(img.getpixel((8,0)), (255,60,30))
        collection = json.loads(Path(r["vectors"]["path"]).read_text())
        self.assertEqual(collection["features"][0]["properties"]["area_pixels"], 12)

    def test_projected_zones(self):
        origin = rasterio.transform.from_origin(500000,3300000,2,3)
        self.values[5,7] = 255
        with rasterio.open(self.root/"grid.tif","w",driver="GTiff",width=8,height=6,
                           count=1,dtype="uint8",crs="EPSG:32650",transform=origin,nodata=255) as ds:
            ds.write(self.values,1)
        polygon = box(500004,3299988,500007,3299997)
        geo = transform_geom("EPSG:32650","EPSG:4326",mapping(polygon))
        (self.root/"zones.json").write_text(json.dumps({"features":[{"geometry":geo,"properties":{"name":"A"}}]}))
        self.src.update(mask="grid.tif",zones="zones.json")
        r = self.run_metrics("zones","pixels","area_m2","vectors")
        self.assertAlmostEqual(r["zones"][0]["changed_m2"],27,places=5)
        self.assertEqual(r["pixels"]["valid"],47)
        self.assertEqual(r["area_m2"]["value"],72)
        self.assertEqual(r["vectors"]["coordinates"],"WGS84")

    def test_empty_and_encoding(self):
        self.src.update(scheme="mci",roi={"bbox":[10,10,20,20]})
        r=self.run_metrics("regions","pixels","roi")
        self.assertEqual(r["regions"]["count"],0)
        self.assertIsNone(r["roi"]["share"])
        self.src.update(scheme="tinycd",fully_valid=False)
        self.assertEqual(self.run_metrics("pixels")["status"],"error")


if __name__ == "__main__":
    unittest.main()
