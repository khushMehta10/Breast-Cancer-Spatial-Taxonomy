#!/usr/bin/env python
"""
Convert every CellViT cells.geojson into a QuPath-ready geojson.

  in :  1_cellvit/output/cellvit_out/<SLIDE>/cells.geojson
  out:  1_cellvit/output/qupath_export/<SLIDE>_cells_qupath.geojson

Repairs invalid CellViT polygons (the cause of QuPath's "Reduction failed"),
keeps the CellViT class colours (with a fixed PanNuke palette as fallback),
and writes a summary CSV of per-slide counts.

Usage:
    python 1_cellvit/scripts/qupath/export_cells_qupath.py                # all slides
    python 1_cellvit/scripts/qupath/export_cells_qupath.py SPA51 SPA52    # selected slides
"""

import csv
import json
import os
import sys
import glob

from shapely.geometry import shape, mapping, MultiPolygon
from shapely.validation import make_valid

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

# ---------------------------------------------------------------- config
IN_ROOT  = str(config.CELLVIT_OUT)
OUT_ROOT = str(config.QUPATH_OUT)

PALETTE = {
    "Neoplastic":   [220,  20,  20],
    "Inflammatory": [  0, 160, 255],
    "Connective":   [  0, 190,  60],
    "Dead":         [240, 220,   0],
    "Epithelial":   [200,   0, 200],
    "Background":   [128, 128, 128],
}
FALLBACK = [[228,26,28], [55,126,184], [77,175,74],
            [152,78,163], [255,127,0], [166,86,40]]


# ---------------------------------------------------------------- helpers
def get_type(props):
    c = props.get("classification")
    if isinstance(c, dict) and c.get("name"):
        return c["name"]
    if isinstance(c, str):
        return c
    for k in ("type", "name", "cell_type", "label"):
        if props.get(k):
            return str(props[k])
    return None


def get_color(props):
    c = props.get("classification")
    if isinstance(c, dict) and isinstance(c.get("color"), (list, tuple)) and len(c["color"]) == 3:
        return list(c["color"])
    return None


def iter_polys(geom):
    """CellViT packs all cells of one class into a single MultiPolygon."""
    if geom.get("type") == "Polygon":
        yield geom["coordinates"]
    elif geom.get("type") == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly


def clean(geom_coords):
    """Repair one polygon; return a valid shapely geom or None."""
    try:
        g = shape({"type": "Polygon", "coordinates": geom_coords})
        if not g.is_valid:
            g = make_valid(g)
        polys, stack = [], [g]
        while stack:
            x = stack.pop()
            if x.geom_type == "Polygon" and x.area > 0:
                polys.append(x)
            elif x.geom_type in ("MultiPolygon", "GeometryCollection"):
                stack.extend(x.geoms)
        if not polys:
            return None
        return polys[0] if len(polys) == 1 else MultiPolygon(polys)
    except Exception:
        return None


# ---------------------------------------------------------------- per slide
def convert(slide):
    src = f"{IN_ROOT}/{slide}/cells.geojson"
    dst = f"{OUT_ROOT}/{slide}_cells_qupath.geojson"

    if not os.path.exists(src):
        return {"slide": slide, "status": "no cells.geojson"}

    gj = json.load(open(src))
    feats = gj["features"] if isinstance(gj, dict) and "features" in gj else gj

    out, dropped, counts, cmap = [], 0, {}, {}
    for ft in feats:
        props = ft.get("properties", {}) or {}
        tname = get_type(props) or "Unclassified"
        src_color = get_color(props)

        for pc in iter_polys(ft.get("geometry") or {}):
            g = clean(pc)
            if g is None:
                dropped += 1
                continue
            if tname not in cmap:
                cmap[tname] = (src_color or PALETTE.get(tname)
                               or FALLBACK[len(cmap) % len(FALLBACK)])
            out.append({
                "type": "Feature",
                "geometry": mapping(g),
                "properties": {
                    "objectType": "detection",
                    "classification": {"name": tname, "color": cmap[tname]},
                },
            })
            counts[tname] = counts.get(tname, 0) + 1

    json.dump({"type": "FeatureCollection", "features": out}, open(dst, "w"))

    row = {"slide": slide, "status": "ok", "cells": len(out), "dropped": dropped}
    row.update({k: counts.get(k, 0) for k in PALETTE if k != "Background"})
    return row


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    if len(sys.argv) > 1:
        slides = sys.argv[1:]
    else:
        slides = sorted(os.path.basename(os.path.dirname(p))
                        for p in glob.glob(f"{IN_ROOT}/*/cells.geojson"))

    print(f"{len(slides)} slides -> {OUT_ROOT}\n")

    rows = []
    for n, slide in enumerate(slides, 1):
        r = convert(slide)
        rows.append(r)
        if r["status"] == "ok":
            print(f"[{n:3d}/{len(slides)}] {slide:10s} {r['cells']:7,d} cells  "
                  f"({r['dropped']} dropped)")
        else:
            print(f"[{n:3d}/{len(slides)}] {slide:10s} SKIPPED - {r['status']}")

    # summary csv
    cols = (["slide", "status", "cells", "dropped"]
            + [k for k in PALETTE if k != "Background"])
    csv_path = f"{OUT_ROOT}/summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\ndone: {len(ok)}/{len(slides)} converted, "
          f"{sum(r['cells'] for r in ok):,} cells total")
    print(f"summary -> {csv_path}")


if __name__ == "__main__":
    main()
