"""
Step 2: measure abundance of immune and cancer cells per window.

For each window and each phenotype (inflammatory, neoplastic), the cell
count is labeled absent (0), low, moderate or high. MODE selects how the
cutoffs are set:

  "quantile" (default)  per-slide quartiles. Among the windows of a slide
        where the phenotype is present, the bottom 25% are low, the middle
        50% moderate and the top 25% high.

  "absolute"  fixed cell counts, identical on every slide: 1 to LOW_MAX
        cells is low, HIGH_MIN or more is high, with HIGH_MIN set to
        Ripley's minimum cell count.

ZERO_HANDLING applies in quantile mode only: zero-count windows are given
their own label and excluded from the quantile computation, since most
slides are strongly zero-inflated for neoplastic and including zeros
collapses q25 and q75 to 0.

Maps carry the tissue boundary, segmented from the H&E thumbnail on HSV
saturation (stained tissue is colored, glass is not).

Reads   step1_window_classes/tables/<SLIDE>_window_classes.csv
        breast/st/<SLIDE>.h5ad  (thumbnail)
Writes  step2_abundance/tables/<SLIDE>_abundance.csv
        step2_abundance/figures/<SLIDE>_abundance.png
        step2_abundance/abundance_summary.csv
        step2_abundance/abundance_thresholds.csv
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
BASE_DIR = config.STATE_OUT
STEP1_DIR = BASE_DIR / "step1_window_classes" / "tables"
ST_DIR = config.BREAST_ST_DIR

OUT_DIR = BASE_DIR / "step2_abundance"
CSV_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
for d in (OUT_DIR, CSV_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SLIDES = "ALL"  # "ALL", or ["SPA51", "TENX68"]
MODE = "quantile"  # "absolute" or "quantile"

# absolute mode: fixed cell-count cutoffs, same on every slide
LOW_MAX = 9  # 1..LOW_MAX          -> low
HIGH_MIN = 50  # HIGH_MIN and above  -> high
# matches Ripley's minimum cell requirement
# in the ATHENA features, so every high window
# has measurable clustering

# quantile mode only
QUARTILE_SCOPE = "slide"  # "slide" or "cohort"
ZERO_HANDLING = "absent"  # "absent" or "low"

MEASURE = "count"  # "count" or "fraction"

DRAW_OUTLINE = True  # segment and draw the tissue boundary
MIN_AREA_FRAC = 0.002  # drop specks below this fraction of the thumbnail
MIN_PTS = 30  # drop contours with fewer vertices

PHENOTYPES = {
    "inflammatory": "immune",
    "neoplastic": "cancer",
}

ABUNDANCE_COLORS = {
    "absent": "#2B2B33",
    "low": "#3E6E8E",
    "moderate": "#2FA37C",
    "high": "#F2C744",
}
ORDER = ["absent", "low", "moderate", "high"]


def tissue_outline(slide):
    """Contours of the tissue in full-resolution slide coordinates.

    Returns [] if the slide has no readable h5ad, so a missing thumbnail
    costs the outline but not the figure.
    """
    if not DRAW_OUTLINE:
        return []
    h5 = ST_DIR / f"{slide}.h5ad"
    if not h5.exists():
        return []

    try:
        import anndata as ad
        from matplotlib.colors import rgb_to_hsv
        from skimage.filters import threshold_otsu
        from skimage.morphology import (
            remove_small_objects,
            remove_small_holes,
            binary_closing,
            disk,
        )
        from skimage.measure import find_contours

        adata = ad.read_h5ad(h5, backed="r")
        lib = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]
        img = np.asarray(lib["images"]["downscaled_fullres"])
        scale = float(lib["scalefactors"]["tissue_downscaled_fullres_scalef"])

        rgb = img[..., :3].astype(float)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0

        sat = rgb_to_hsv(rgb)[..., 1]
        mask = sat > threshold_otsu(sat)

        min_area = int(MIN_AREA_FRAC * mask.size)
        mask = binary_closing(mask, disk(3))
        mask = remove_small_holes(mask, area_threshold=min_area)
        mask = remove_small_objects(mask, min_size=min_area)

        cs = [c for c in find_contours(mask.astype(float), 0.5) if len(c) >= MIN_PTS]
        # find_contours gives (row, col) = (y, x); rescale to full resolution
        return [np.column_stack([c[:, 1] / scale, c[:, 0] / scale]) for c in cs]
    except Exception as ex:
        print(f"   outline unavailable ({type(ex).__name__})")
        return []


def assign_absolute(values):
    """Fixed cell-count cutoffs, identical on every slide."""
    out = np.empty(len(values), dtype=object)
    out[values == 0] = "absent"
    out[(values > 0) & (values <= LOW_MAX)] = "low"
    out[(values > LOW_MAX) & (values < HIGH_MIN)] = "moderate"
    out[values >= HIGH_MIN] = "high"
    return out


def assign_abundance(values, q25, q75, zero_handling):
    """Label each window given precomputed thresholds."""
    out = np.empty(len(values), dtype=object)
    present = values > 0

    if zero_handling == "absent":
        out[~present] = "absent"
    else:
        out[~present] = "low"

    out[present & (values <= q25)] = "low"
    out[present & (values > q25) & (values <= q75)] = "moderate"
    out[present & (values > q75)] = "high"
    return out


def thresholds(values, zero_handling):
    """q25 and q75 for a set of window values."""
    v = values[values > 0] if zero_handling == "absent" else values
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 25)), float(np.percentile(v, 75))


# ============================================================
# LOAD
# ============================================================
if SLIDES == "ALL":
    files = sorted(glob.glob(str(STEP1_DIR / "*_window_classes.csv")))
else:
    files = [str(STEP1_DIR / f"{s}_window_classes.csv") for s in SLIDES]

if not files:
    raise SystemExit(
        f"No step 1 tables found in {STEP1_DIR}. Run classify_windows.py first."
    )

# Cohort: slides with a breast h5ad, minus config.EXCLUDED_SLIDES
frames, skipped = {}, []
for path in files:
    slide = os.path.basename(path).replace("_window_classes.csv", "")
    if not (ST_DIR / f"{slide}.h5ad").exists() or slide in config.EXCLUDED_SLIDES:
        skipped.append(slide)
        continue
    frames[slide] = pd.read_csv(path)

print(f"loaded {len(frames)} slides, skipped {len(skipped)}")
if not frames:
    raise SystemExit("No cohort slides found in the step 1 tables.")

# cohort-wide thresholds, if requested (quantile mode only)
cohort_thr = {}
if MODE == "quantile" and QUARTILE_SCOPE == "cohort":
    allrows = pd.concat(frames.values(), ignore_index=True)
    for pheno in PHENOTYPES:
        col = f"{pheno}_{MEASURE}"
        cohort_thr[pheno] = thresholds(allrows[col].fillna(0).to_numpy(), ZERO_HANDLING)
        print(
            f"cohort {pheno}: q25={cohort_thr[pheno][0]:.3f}  q75={cohort_thr[pheno][1]:.3f}"
        )


# ============================================================
# RUN
# ============================================================
summary_rows, threshold_rows = [], []

for i, (slide, df) in enumerate(sorted(frames.items()), 1):
    row = {
        "slide_id": slide,
        "technology": df["technology"].iloc[0] if "technology" in df.columns else "",
        "n_windows": len(df),
    }
    thr_row = {"slide_id": slide}

    print(f"\n[{i}/{len(frames)}] {slide}: {len(df)} windows", flush=True)

    for pheno, short in PHENOTYPES.items():
        col = f"{pheno}_{MEASURE}"
        vals = df[col].fillna(0).to_numpy(float)

        n_present = int((vals > 0).sum())

        if MODE == "absolute":
            df[f"{short}_abundance"] = assign_absolute(vals)
            q25, q75 = float(LOW_MAX), float(HIGH_MIN)
            counts = pd.Series(df[f"{short}_abundance"]).value_counts()
            parts = "  ".join(f"{lab}={int(counts.get(lab, 0))}" for lab in ORDER)
            print(
                f"   {short:<7} present={n_present:<5} "
                f"low<={LOW_MAX} high>={HIGH_MIN}   {parts}"
            )
        else:
            if QUARTILE_SCOPE == "cohort":
                q25, q75 = cohort_thr[pheno]
            else:
                q25, q75 = thresholds(vals, ZERO_HANDLING)

            if np.isnan(q25):
                df[f"{short}_abundance"] = "absent"
                print(f"   {short:<7} no windows contain this phenotype")
            else:
                df[f"{short}_abundance"] = assign_abundance(
                    vals, q25, q75, ZERO_HANDLING
                )
                counts = pd.Series(df[f"{short}_abundance"]).value_counts()
                parts = "  ".join(f"{lab}={int(counts.get(lab, 0))}" for lab in ORDER)
                print(
                    f"   {short:<7} present={n_present:<5} "
                    f"q25={q25:8.2f} q75={q75:8.2f}   {parts}"
                )

        thr_row[f"{short}_low_max"] = q25
        thr_row[f"{short}_high_min"] = q75
        thr_row[f"{short}_n_present"] = n_present

        counts = pd.Series(df[f"{short}_abundance"]).value_counts()
        for lab in ORDER:
            row[f"{short}_{lab}"] = int(counts.get(lab, 0))

    summary_rows.append(row)
    threshold_rows.append(thr_row)

    # ---- save ----
    df.to_csv(CSV_DIR / f"{slide}_abundance.csv", index=False)

    outline = tissue_outline(slide)

    # ---- two-panel map ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, (pheno, short) in zip(axes, PHENOTYPES.items()):
        ax.set_facecolor("#101018")
        rects, colors = [], []
        for r in df.itertuples(index=False):
            lab = getattr(r, f"{short}_abundance")
            colors.append(ABUNDANCE_COLORS.get(lab, "#FFFFFF"))
            rects.append(
                patches.Rectangle((r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin)
            )
        ax.add_collection(
            PatchCollection(rects, facecolors=colors, edgecolors="none", linewidths=0)
        )
        for seg in outline:
            ax.plot(seg[:, 0], seg[:, 1], color="white", linewidth=1.0, zorder=5)

        pad = 0.05 * (df["xmax"].max() - df["xmin"].min())
        ax.set_xlim(df["xmin"].min() - pad, df["xmax"].max() + pad)
        ax.set_ylim(df["ymin"].min() - pad, df["ymax"].max() + pad)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title(f"{short} abundance")

        counts = pd.Series(df[f"{short}_abundance"]).value_counts()
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=9,
                markerfacecolor=ABUNDANCE_COLORS[lab],
                markeredgecolor="none",
                label=f"{lab} (n={int(counts.get(lab, 0))})",
            )
            for lab in ORDER
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=True)

    scope = f"{QUARTILE_SCOPE} quartiles" if MODE == "quantile" else "absolute cutoffs"
    fig.suptitle(f"{slide}: abundance by window ({scope}, {MEASURE})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{slide}_abundance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================
s = pd.DataFrame(summary_rows)
s.to_csv(OUT_DIR / "abundance_summary.csv", index=False)
pd.DataFrame(threshold_rows).to_csv(OUT_DIR / "abundance_thresholds.csv", index=False)

print("\n" + "=" * 60)
print("COHORT TOTALS")
print("=" * 60)
total = int(s["n_windows"].sum())
print(f"slides: {len(s)}   windows: {total:,}")
if MODE == "absolute":
    print(
        f"absolute cutoffs: low 1-{LOW_MAX}, moderate {LOW_MAX+1}-{HIGH_MIN-1}, "
        f"high >={HIGH_MIN}   measure: {MEASURE}"
    )
else:
    print(
        f"quantile mode: {QUARTILE_SCOPE}   zeros: {ZERO_HANDLING}   "
        f"measure: {MEASURE}"
    )
for short in PHENOTYPES.values():
    print(f"\n  {short}")
    for lab in ORDER:
        n = int(s[f"{short}_{lab}"].sum())
        print(f"    {lab:<9} {n:>7,}  ({100*n/total:5.1f}%)")

print("\ndone ->", OUT_DIR)
