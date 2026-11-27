#!/usr/bin/env python
"""
Make QuPath spot overlays for the three platforms.

  out: 1_cellvit/output/qupath_export/<SLIDE>_spots_qupath.geojson

Spot radius is half the spot diameter (um) divided by the pixel size (um/px),
both from the HEST metadata, so the circles are drawn at their true physical size on each slide.

Usage:
    python 1_cellvit/scripts/qupath/export_spots_qupath.py
"""

import json
import os

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

ST_DIR = str(config.BREAST_ST_DIR)
OUT_ROOT = str(config.QUPATH_OUT)

# slide: (platform, spot_diameter_um, pixel_size_um)   from HEST metadata
SLIDES = {
    "SPA51": ("Spatial Transcriptomics", 100.0, 0.6861319494),
    "NCBI776": ("Visium", 55.0, 0.5475531951),
    "NCBI783": ("Xenium", None, 0.2739598560),  # no spots
}

COLOR = [0, 0, 0]  # black outline
NPTS = 32  # vertices per circle


def circle(cx, cy, r, n=NPTS):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]
    return np.vstack([pts, pts[0]]).round(1).tolist()


def feature(ring, name):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "objectType": "annotation",
            "classification": {"name": name, "color": COLOR},
        },
    }


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for slide, (platform, diam_um, mpp) in SLIDES.items():
        h5 = f"{ST_DIR}/{slide}.h5ad"
        if not os.path.exists(h5):
            print(f"{slide:8s} SKIP - no h5ad")
            continue

        try:
            import anndata as ad

            spots = np.asarray(
                ad.read_h5ad(h5, backed="r").obsm["spatial"], dtype=float
            )
        except Exception:
            import h5py

            with h5py.File(h5, "r") as f:
                spots = f["obsm/spatial"][:].astype(float)

        if diam_um is None:
            # Xenium: subcellular, no real spot. Draw small markers so the
            # geometry is still visible next to the other platforms.
            r = 10.0
            note = "no true spots (subcellular) - 10px markers"
        else:
            r = (diam_um / 2.0) / mpp
            note = f"{diam_um:.0f}um spot -> r={r:.1f}px"

        feats = [feature(circle(cx, cy, r), "Spot") for cx, cy in spots]
        dst = f"{OUT_ROOT}/{slide}_spots_qupath.geojson"
        json.dump({"type": "FeatureCollection", "features": feats}, open(dst, "w"))

        print(f"{slide:8s} {platform:24s} {len(spots):6,d} spots   {note}")
        print(f"         -> {dst}")


if __name__ == "__main__":
    main()
