"""
Step 1: classify windows by cell composition.

  Class 1  Neither      NI = 0 and NC = 0
  Class 2  Immune only  NI > 0 and NC = 0
  Class 3  Cancer only  NC > 0 and NI = 0
  Class 4  Both         NI > 0 and NC > 0

NI = inflammatory_count, NC = neoplastic_count. Connective, epithelial and
dead are not used: stroma is present in nearly every window, so requiring it
to be absent selects near-empty windows rather than pure phenotype regions.
Every window therefore falls into exactly one class.

STRICT_PURE additionally flags the rare windows containing a single phenotype
and nothing else, as a boolean column rather than a separate class.

Maps are drawn with x horizontal and y vertical, with the tissue outline
segmented from the H&E thumbnail (saturation threshold) drawn on top.

The cohort is every slide with a breast h5ad, minus the slides listed in
config.EXCLUDED_SLIDES.

Reads   athena_2/<SLIDE>_athena_features.csv
        breast/st/<SLIDE>.h5ad   (thumbnail, for the tissue outline)
Writes  step1_window_classes/tables/<SLIDE>_window_classes.csv
        step1_window_classes/figures/<SLIDE>_window_classes.png
        step1_window_classes/window_class_summary.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from pathlib import Path
import glob, os

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

# ============================================================
# CONFIG
# ============================================================
ATHENA2_DIR = config.ATHENA2_DIR
ST_DIR      = config.BREAST_ST_DIR
BASE_DIR    = config.STATE_OUT

OUT_DIR = BASE_DIR / "step1_window_classes"
CSV_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
for d in (OUT_DIR, CSV_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SLIDES       = "ALL"    # "ALL", or a list like ["SPA51", "TENX68"]
STRICT_PURE  = True     # flag windows containing one phenotype and nothing else
DRAW_OUTLINE = True     # segment and draw the tissue boundary

# tissue segmentation
MIN_AREA_FRAC = 0.002   # drop specks below this fraction of the thumbnail
MIN_PTS       = 30      # drop contours with fewer vertices

CLASS_INFO = {
    1: ("Neither",      "#3C4043"),
    2: ("Immune only",  "#22DD4D"),
    3: ("Cancer only",  "#FF0000"),
    4: ("Both",         "#8E44AD"),
}

COUNT_COLS = ["inflammatory_count", "neoplastic_count",
              "connective_count", "epithelial_count", "dead_count"]


def classify(df):
    """Class by immune and cancer presence alone."""
    ni = df["inflammatory_count"].fillna(0).to_numpy()
    nc = df["neoplastic_count"].fillna(0).to_numpy()

    out = np.zeros(len(df), dtype=int)
    out[(ni == 0) & (nc == 0)] = 1
    out[(ni > 0)  & (nc == 0)] = 2
    out[(nc > 0)  & (ni == 0)] = 3
    out[(ni > 0)  & (nc > 0)]  = 4
    return out


def pure_flag(df):
    """True where exactly one phenotype is present in the window."""
    present = [df[c].fillna(0).to_numpy() > 0 for c in COUNT_COLS if c in df.columns]
    return np.sum(present, axis=0) == 1


def tissue_outline(slide):
    """Contours of the tissue, in full-resolution slide coordinates.

    Segments the H&E thumbnail on HSV saturation: stained tissue is coloured,
    glass is white or grey. Returns [] if the slide has no readable h5ad.
    """
    h5 = ST_DIR / f"{slide}.h5ad"
    if not h5.exists():
        return []

    try:
        import anndata as ad
        from matplotlib.colors import rgb_to_hsv
        from skimage.filters import threshold_otsu
        from skimage.morphology import (remove_small_objects, remove_small_holes,
                                        binary_closing, disk)
        from skimage.measure import find_contours

        adata = ad.read_h5ad(h5, backed="r")
        lib   = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]
        img   = np.asarray(lib["images"]["downscaled_fullres"])
        scale = float(lib["scalefactors"]["tissue_downscaled_fullres_scalef"])

        rgb = img[..., :3].astype(float)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0

        sat  = rgb_to_hsv(rgb)[..., 1]
        mask = sat > threshold_otsu(sat)

        min_area = int(MIN_AREA_FRAC * mask.size)
        mask = binary_closing(mask, disk(3))
        mask = remove_small_holes(mask, area_threshold=min_area)
        mask = remove_small_objects(mask, min_size=min_area)

        cs = [c for c in find_contours(mask.astype(float), 0.5) if len(c) >= MIN_PTS]
        # find_contours gives (row, col) = (y, x); rescale to full resolution
        return [np.column_stack([c[:, 1] / scale, c[:, 0] / scale]) for c in cs]
    except Exception as e:
        print(f"   outline unavailable ({type(e).__name__})")
        return []


# ============================================================
# RUN
# ============================================================
if SLIDES == "ALL":
    files = sorted(glob.glob(str(ATHENA2_DIR / "*_athena_features.csv")))
else:
    files = [str(ATHENA2_DIR / f"{s}_athena_features.csv") for s in SLIDES]

# Cohort: slides with a breast h5ad, minus config.EXCLUDED_SLIDES
files = [f for f in files
         if (ST_DIR / os.path.basename(f).replace("_athena_features.csv", ".h5ad")).exists()
         and os.path.basename(f).replace("_athena_features.csv", "") not in config.EXCLUDED_SLIDES]

summary_rows = []

for i, path in enumerate(files, 1):
    slide = os.path.basename(path).replace("_athena_features.csv", "")
    if not os.path.exists(path):
        print(f"SKIP {slide} (no features file)")
        continue

    df = pd.read_csv(path)
    df["window_class"] = classify(df)
    df["window_class_label"] = df["window_class"].map(
        {k: v[0] for k, v in CLASS_INFO.items()}
    )
    if STRICT_PURE:
        df["single_phenotype"] = pure_flag(df)

    counts = df["window_class"].value_counts().reindex([1, 2, 3, 4], fill_value=0)
    n_unassigned = int((df["window_class"] == 0).sum())

    print(f"\n[{i}/{len(files)}] {slide}: {len(df)} windows", flush=True)
    for k in [1, 2, 3, 4]:
        n = int(counts[k])
        print(f"   class {k}  {CLASS_INFO[k][0]:<14} {n:>6}  ({100*n/len(df):5.1f}%)")
    if STRICT_PURE:
        n_pure = int(df["single_phenotype"].sum())
        print(f"   {'single-phenotype windows':<28} {n_pure:>6}  ({100*n_pure/len(df):5.1f}%)")
    if n_unassigned:
        print(f"   WARNING: {n_unassigned} windows unassigned")

    # ---- save assignments ----
    keep = ["slide_id", "window_id", "technology", "grid_row", "grid_col",
            "xmin", "xmax", "ymin", "ymax",
            "inflammatory_count", "neoplastic_count",
            "connective_count", "epithelial_count", "dead_count",
            "inflammatory_fraction", "neoplastic_fraction",
            "connective_fraction", "epithelial_fraction", "dead_fraction",
            "window_class", "window_class_label"]
    if STRICT_PURE:
        keep.append("single_phenotype")
    keep = [c for c in keep if c in df.columns]

    csv_out = CSV_DIR / f"{slide}_window_classes.csv"
    df[keep].to_csv(csv_out, index=False)

    row = {"slide_id": slide,
           "technology": df["technology"].iloc[0] if "technology" in df.columns else "",
           "n_windows": len(df)}
    for k in [1, 2, 3, 4]:
        row[f"class{k}_{CLASS_INFO[k][0].replace(' ', '_').lower()}"] = int(counts[k])
    if STRICT_PURE:
        row["single_phenotype"] = int(df["single_phenotype"].sum())
    summary_rows.append(row)

    # ---- map: x horizontal, y vertical ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"{slide}: windows by cell composition")
    ax.set_facecolor("#101018")

    rects, colours = [], []
    for r in df.itertuples(index=False):
        colours.append(CLASS_INFO.get(r.window_class, ("", "#FFFFFF"))[1])
        rects.append(patches.Rectangle((r.xmin, r.ymin),
                                       r.xmax - r.xmin, r.ymax - r.ymin))
    ax.add_collection(PatchCollection(rects, facecolors=colours,
                                      edgecolors="none", linewidths=0))

    if DRAW_OUTLINE:
        for seg in tissue_outline(slide):
            ax.plot(seg[:, 0], seg[:, 1], color="white", linewidth=1.2, zorder=5)

    pad = 0.05 * (df["xmax"].max() - df["xmin"].min())
    ax.set_xlim(df["xmin"].min() - pad, df["xmax"].max() + pad)
    ax.set_ylim(df["ymin"].min() - pad, df["ymax"].max() + pad)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=10,
               markerfacecolor=CLASS_INFO[k][1], markeredgecolor="none",
               label=f"{k}. {CLASS_INFO[k][0]} (n={int(counts[k])})")
        for k in [1, 2, 3, 4]
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    fig_out = FIG_DIR / f"{slide}_window_classes.png"
    plt.tight_layout()
    plt.savefig(fig_out, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---- cohort summary ----
if summary_rows:
    s = pd.DataFrame(summary_rows)
    s.to_csv(OUT_DIR / "window_class_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("COHORT TOTALS")
    print("=" * 60)
    total = int(s["n_windows"].sum())
    print(f"slides: {len(s)}   windows: {total:,}")
    for k in [1, 2, 3, 4]:
        col = f"class{k}_{CLASS_INFO[k][0].replace(' ', '_').lower()}"
        n = int(s[col].sum())
        print(f"   class {k}  {CLASS_INFO[k][0]:<14} {n:>7,}  ({100*n/total:5.1f}%)")
    if STRICT_PURE:
        n = int(s["single_phenotype"].sum())
        print(f"   {'single-phenotype windows':<28} {n:>7,}  ({100*n/total:5.1f}%)")

print("\ndone ->", OUT_DIR)
