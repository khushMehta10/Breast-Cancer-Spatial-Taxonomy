"""
One per-cell table per slide from the CellViT output: centroid, phenotype and
contour morphology (area, perimeter, circularity, solidity, eccentricity).

  in : 1_cellvit/output/cellvit_out/<SLIDE>/{cell_detection.json, cells.geojson}
  out: 2_athena/output/tables/<SLIDE>_cells.csv
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, MultiPolygon
from skimage.draw import polygon as draw_polygon
from skimage.measure import regionprops


def clean_contour(contour):
    points = np.asarray(contour, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(points[1:] != points[:-1], axis=1)
    points = points[keep]
    if len(points) < 3:
        return None
    return points


def contour_to_local_mask(points, padding=2):
    x, y = points[:, 0], points[:, 1]
    x_min = int(np.floor(x.min())) - padding
    y_min = int(np.floor(y.min())) - padding
    x_max = int(np.ceil(x.max())) + padding
    y_max = int(np.ceil(y.max())) + padding
    width, height = x_max - x_min + 1, y_max - y_min + 1
    if width <= 0 or height <= 0:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    rr, cc = draw_polygon(y - y_min, x - x_min, shape=mask.shape)
    mask[rr, cc] = 1
    return mask


def calculate_morphology(contour):
    points = clean_contour(contour)
    if points is None:
        return None
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    if not isinstance(poly, Polygon):
        return None
    area = float(poly.area)
    perimeter = float(poly.length)
    if area <= 0 or perimeter <= 0:
        return None
    circularity = 4.0 * np.pi * area / (perimeter**2)
    hull_area = float(poly.convex_hull.area)
    solidity = area / hull_area if hull_area > 0 else np.nan
    eccentricity = np.nan
    mask = contour_to_local_mask(points)
    if mask is not None and mask.any():
        regions = regionprops(mask)
        if regions:
            eccentricity = float(regions[0].eccentricity)
    return {
        "contour_centroid_x": float(poly.centroid.x),
        "contour_centroid_y": float(poly.centroid.y),
        "area_px2": area,
        "perimeter_px": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
    }


def build_cellvit_dataframe(cell_detection_path, cells_geojson_path):
    cell_detection_path = Path(cell_detection_path)
    cells_geojson_path = Path(cells_geojson_path)
    with cell_detection_path.open("r") as file:
        detection_data = json.load(file)
    type_map = {int(t): c for t, c in detection_data["type_map"].items()}
    detection_rows = []
    for n, cell in enumerate(detection_data["cells"]):
        cx, cy = cell["centroid"]
        tid = int(cell["type"])
        detection_rows.append(
            {
                "cell_id": f"cell_{n:06d}",
                "x": float(cx),
                "y": float(cy),
                "cell_type_id": tid,
                "cell_type": type_map.get(tid, "Unknown"),
            }
        )
    obs_df = pd.DataFrame(detection_rows)
    with cells_geojson_path.open("r") as file:
        geojson_features = json.load(file)
    morphology_rows = []
    for feature in geojson_features:
        cell_type = feature["properties"]["classification"]["name"]
        geometry = feature["geometry"]
        if geometry["type"] != "MultiPolygon":
            continue
        for polygon in geometry["coordinates"]:
            if not polygon:
                continue
            m = calculate_morphology(polygon[0])
            if m is None:
                continue
            morphology_rows.append({"cell_type": cell_type, **m})
    morphology_df = pd.DataFrame(morphology_rows)
    morphology_df["cell_id"] = None
    morphology_df["matching_distance_px"] = np.nan
    for cell_type in morphology_df["cell_type"].unique():
        detections = obs_df[obs_df["cell_type"] == cell_type].copy()
        contours = morphology_df[morphology_df["cell_type"] == cell_type].copy()
        if detections.empty or contours.empty:
            continue
        tree = cKDTree(detections[["x", "y"]].to_numpy())
        distances, positions = tree.query(
            contours[["contour_centroid_x", "contour_centroid_y"]].to_numpy(), k=1
        )
        morphology_df.loc[contours.index, "cell_id"] = detections.iloc[positions][
            "cell_id"
        ].to_numpy()
        morphology_df.loc[contours.index, "matching_distance_px"] = distances
    morph_merge = morphology_df[
        [
            "cell_id",
            "area_px2",
            "perimeter_px",
            "circularity",
            "solidity",
            "eccentricity",
        ]
    ]
    final_df = obs_df.merge(
        morph_merge, on="cell_id", how="left", validate="one_to_one"
    )
    final_df = final_df[
        [
            "cell_id",
            "x",
            "y",
            "cell_type_id",
            "cell_type",
            "area_px2",
            "perimeter_px",
            "circularity",
            "solidity",
            "eccentricity",
        ]
    ]
    print("Detected cells:", len(obs_df))
    print("Extracted contours:", len(morphology_df))
    print("Final dataframe shape:", final_df.shape)
    print("Unmatched contours:", morphology_df["cell_id"].isna().sum())
    print("Duplicate matches:", morphology_df["cell_id"].dropna().duplicated().sum())
    print("Missing morphology rows:", final_df["area_px2"].isna().sum())
    return final_df


import os, glob

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

IN_ROOT = str(config.CELLVIT_OUT)
OUT_ROOT = str(config.ATHENA_OUT / "tables")
os.makedirs(OUT_ROOT, exist_ok=True)

slides = sorted(
    os.path.basename(os.path.dirname(p))
    for p in glob.glob(f"{IN_ROOT}/*/cells.geojson")
)
print(f"{len(slides)} slides\n", flush=True)

for i, slide in enumerate(slides, 1):
    out = f"{OUT_ROOT}/{slide}_cells.csv"
    if os.path.exists(out):
        print(f"[{i}/{len(slides)}] {slide} - already done, skipping", flush=True)
        continue
    print(f"===== [{i}/{len(slides)}] {slide} =====", flush=True)
    try:
        df = build_cellvit_dataframe(
            f"{IN_ROOT}/{slide}/cell_detection.json", f"{IN_ROOT}/{slide}/cells.geojson"
        )
        df.to_csv(out, index=False)
        print("wrote", out, "\n", flush=True)
    except Exception as e:
        print("FAILED:", e, "\n", flush=True)
