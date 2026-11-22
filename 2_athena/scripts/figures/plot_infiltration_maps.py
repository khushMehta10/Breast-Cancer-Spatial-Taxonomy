"""
Infiltration maps -- window boxes at their true extent, with tissue outline.

PART 1  Segment tissue from the H&E thumbnail (saturation threshold), trace
        contours, rescale to full-resolution slide coordinates.

PART 2  Draw one rectangle per WINDOW at its real xmin/xmax/ymin/ymax extent,
        coloured by the infiltration score, with the tissue outline on top.

Axes are transposed: Y horizontal, X vertical,
X increasing downward. Horizontal colorbar below. All 12 directed pairs.

Correctness guards (toggleable):
  CLIP_PCTILE    colour ceiling percentile, so outliers don't flatten the map
  MIN_HOMOTYPIC  windows with too few source-source edges -> undefined
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import rgb_to_hsv
from skimage.filters import threshold_otsu
from skimage.morphology import (remove_small_objects, remove_small_holes,
                                binary_closing, disk)
from skimage.measure import find_contours
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

# ============================================================
# CONFIG
# ============================================================
SLIDE_ID   = "SPA51"

ST_DIR     = config.BREAST_ST_DIR
ATHENA_DIR = config.ATHENA_DIR
FIG_DIR    = config.ATHENA_OUT / "figures" / "infiltration"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELL_TYPES = ["connective", "epithelial", "inflammatory", "neoplastic"]

CLIP_PCTILE   = 95      # colour ceiling percentile; None = raw max
MIN_HOMOTYPIC = 10      # min source-source edges to trust a window; 0 = off

MIN_AREA_FRAC = 0.002   # tissue segmentation: drop specks below this fraction
MIN_PTS       = 30      # drop contours with fewer vertices

cmap  = plt.cm.viridis
FLOOR = cmap(0.0)       # background and undefined windows


# ============================================================
# PART 1 -- TISSUE OUTLINE
# ============================================================
print("=" * 60)
print("PART 1: tissue segmentation")
print("=" * 60)

adata = ad.read_h5ad(ST_DIR / f"{SLIDE_ID}.h5ad")
lib   = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]
img   = np.asarray(lib["images"]["downscaled_fullres"])
scale = float(lib["scalefactors"]["tissue_downscaled_fullres_scalef"])
print(f"thumbnail {img.shape[1]}x{img.shape[0]}, scale {scale:.6f}")

rgb = img[..., :3].astype(float)
if rgb.max() > 1.5:
    rgb = rgb / 255.0

sat  = rgb_to_hsv(rgb)[..., 1]         # stained tissue is coloured; glass is not
thr  = threshold_otsu(sat)
mask = sat > thr

min_area = int(MIN_AREA_FRAC * mask.size)
mask = binary_closing(mask, disk(3))
mask = remove_small_holes(mask, area_threshold=min_area)
mask = remove_small_objects(mask, min_size=min_area)

contours = [c for c in find_contours(mask.astype(float), 0.5) if len(c) >= MIN_PTS]
contours.sort(key=len, reverse=True)

# find_contours returns (row, col) = (y, x); rescale to full-resolution
segments = [np.column_stack([c[:, 1] / scale, c[:, 0] / scale]) for c in contours]
print(f"contours: {len(segments)}")


# ============================================================
# PART 2 -- INFILTRATION MAPS
# ============================================================
print()
print("=" * 60)
print("PART 2: infiltration maps (window boxes)")
print("=" * 60)

features = pd.read_csv(ATHENA_DIR / "features_df" / f"{SLIDE_ID}_features.csv")
windows  = pd.read_csv(ATHENA_DIR / "window_df"   / f"{SLIDE_ID}_windows.csv")

# join scores to the window geometry (xmin/xmax/ymin/ymax)
df = features.merge(
    windows[["window_id", "xmin", "xmax", "ymin", "ymax"]],
    on="window_id", how="left",
)
df = df.dropna(subset=["xmin", "xmax", "ymin", "ymax"])

box_w = float(np.median(df["xmax"] - df["xmin"]))
box_h = float(np.median(df["ymax"] - df["ymin"]))
print(f"windows: {len(df):,}   median box: {box_w:.0f} x {box_h:.0f} px")

pairs = [(a, b) for a in CELL_TYPES for b in CELL_TYPES if a != b]

for src, tgt in pairs:
    col = f"infiltration_{src}_to_{tgt}"
    if col not in df.columns:
        print(f"SKIP {col} (not in features)")
        continue

    vals = df[col].to_numpy(float).copy()

    # denominator guard
    homo_col = f"homotypic_edges_{src}"
    n_dropped = 0
    if MIN_HOMOTYPIC and homo_col in df.columns:
        weak = df[homo_col].fillna(0).to_numpy() < MIN_HOMOTYPIC
        n_dropped = int((weak & ~np.isnan(vals)).sum())
        vals[weak] = np.nan

    defined = vals[~np.isnan(vals)]
    if defined.size == 0:
        print(f"SKIP {col} (no trustworthy windows)")
        continue

    vmax = np.percentile(defined, CLIP_PCTILE) if CLIP_PCTILE else defined.max()
    vmax = max(float(vmax), 1e-9)
    norm = plt.Normalize(0.0, vmax)

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_title(f"Slide {SLIDE_ID}")
    ax.set_facecolor(FLOOR)

    # one rectangle per window, at its true extent, transposed -> (y, x)
    for row, val in zip(df.itertuples(index=False), vals):
        face = FLOOR if np.isnan(val) else cmap(norm(val))
        ax.add_patch(patches.Rectangle(
            (row.ymin, row.xmin),
            row.ymax - row.ymin,          # width  along Y (horizontal)
            row.xmax - row.xmin,          # height along X (vertical)
            linewidth=0, facecolor=face,
        ))

    # tissue outline on top, transposed to match
    for seg in segments:
        ax.plot(seg[:, 1], seg[:, 0], color="black", linewidth=1.4, zorder=5)

    pad = max(box_w, box_h) * 1.5
    ax.set_xlim(df["ymin"].min() - pad, df["ymax"].max() + pad)
    ax.set_ylim(df["xmin"].min() - pad, df["xmax"].max() + pad)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xlabel("Y coordinate")
    ax.set_ylabel("X coordinate")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.046, pad=0.10)
    cbar.set_label(col)

    out = FIG_DIR / f"{SLIDE_ID}_{col}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"{col}: {defined.size} windows, vmax(p{CLIP_PCTILE})={vmax:.2f}"
          + (f", {n_dropped} dropped (<{MIN_HOMOTYPIC} homotypic)" if n_dropped else "")
          + f"  -> {out.name}")

print("\ndone ->", FIG_DIR)
