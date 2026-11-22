"""
QuPath annotation boxes of 1024 px around every spot, for every slide.

  out: 1_cellvit/output/qupath_export/boxes_1024/<SLIDE>_boxes_1024px.geojson
"""

import os, json, glob, numpy as np, anndata as ad

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

OUT  = str(config.QUPATH_OUT / "boxes_1024")
SIDE = 1024
os.makedirs(OUT, exist_ok=True)

def square(cx, cy, s):
    h=s/2.0; return [[cx-h,cy-h],[cx+h,cy-h],[cx+h,cy+h],[cx-h,cy+h],[cx-h,cy-h]]

slides = sorted(glob.glob(str(config.BREAST_ST_DIR / "*.h5ad")))
print(len(slides), "slides")

for h5 in slides:
    sid = os.path.basename(h5).replace(".h5ad","")
    try:
        spots = np.asarray(ad.read_h5ad(h5, backed="r").obsm["spatial"], dtype=float)
    except Exception as e:
        print("[skip]", sid, e); continue
    feats=[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[square(cx,cy,SIDE)]},
            "properties":{"objectType":"annotation","classification":{"name":"box_1024","color":[255,255,0]}}}
           for cx,cy in spots]
    json.dump({"type":"FeatureCollection","features":feats}, open(f"{OUT}/{sid}_boxes_1024px.geojson","w"))
    print(f"{sid}: {len(spots)} boxes")
print("done ->", OUT)


