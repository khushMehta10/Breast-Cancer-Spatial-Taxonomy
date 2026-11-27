"""
QuPath annotation boxes of 512, 1024 and 2048 px, centered on the spot nearest
the center of the tissue, for a few selected slides.

  out: 1_cellvit/output/qupath_export/<SLIDE>_onebox_<SIZE>px.geojson
"""

import json, os
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

ST_DIR = str(config.BREAST_ST_DIR)
OUT_ROOT = str(config.QUPATH_OUT)

SLIDES = ["SPA51", "NCBI776", "NCBI783"]
SIZES = [512, 1024, 2048]
COLOR = [255, 255, 0]

os.makedirs(OUT_ROOT, exist_ok=True)

for slide in SLIDES:
    h5 = f"{ST_DIR}/{slide}.h5ad"
    if not os.path.exists(h5):
        print(f"{slide:8s} SKIP - no h5ad")
        continue

    try:
        import anndata as ad

        spots = np.asarray(ad.read_h5ad(h5, backed="r").obsm["spatial"], dtype=float)
    except Exception:
        import h5py

        with h5py.File(h5, "r") as f:
            spots = f["obsm/spatial"][:].astype(float)

    i = int(np.argmin(((spots - spots.mean(0)) ** 2).sum(1)))
    cx, cy = spots[i]

    for size in SIZES:
        h = size / 2.0
        ring = [
            [cx - h, cy - h],
            [cx + h, cy - h],
            [cx + h, cy + h],
            [cx - h, cy + h],
            [cx - h, cy - h],
        ]
        feat = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "objectType": "annotation",
                "classification": {"name": f"box_{size}", "color": COLOR},
            },
        }
        dst = f"{OUT_ROOT}/{slide}_onebox_{size}px.geojson"
        json.dump({"type": "FeatureCollection", "features": [feat]}, open(dst, "w"))
        print(f"{slide:8s} {size:5d}px  spot #{i} at ({cx:.0f}, {cy:.0f})  -> {dst}")
