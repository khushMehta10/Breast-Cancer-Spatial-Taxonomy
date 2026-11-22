"""
Step 6: engagement of immune and cancer with the connective and epithelial
compartments,
plus tumour and immune burden.

Measures immune interactions with connective and epithelial cells, the
same for cancer, and the same again for infiltration. Connective is
taken as the tumour microenvironment proper. Then tumour burden, immune
burden, and the tumour-immune niche fraction per slide.

Eight per-window measures, each flagged high or low: interaction at zero,
infiltration at the within-slide median over windows where it is defined:

  interaction   immune -> connective     immune -> epithelial
                cancer -> connective     cancer -> epithelial
  infiltration  immune -> connective     immune -> epithelial
                cancer -> connective     cancer -> epithelial

and a summary label per phenotype:

  tme_engaged        high interaction with connective
  epithelium_engaged high interaction with epithelial
  both_engaged       high with both
  low_engagement     high with neither
  not_present        the phenotype is absent from the window

Burden, per slide:
  tumour_burden   share of that slide's cells that are neoplastic
  immune_burden   share that are inflammatory
  niche_fraction  share of both-phenotype windows labelled niche in step 5
  tumour and immune burden are also reported as window-level means

--------------------------------------------------------------------------
DESIGN CHOICES

1. HIGH / LOW depends on whether the quantity has a meaningful zero.

   INTERACTION is enrichment against a permutation null, so zero is the
   value expected by chance and the sign is meaningful: high means more
   contact than chance. Splitting it at a slide median instead would make
   "high" mean only "least negative here", since cross-type interaction is
   negative in most windows.

   INFILTRATION is a ratio of source-target to source-source edges, bounded
   below by zero rather than centred on it. Zero means no cross-type contact
   at all, not chance-level contact, so it takes a within-slide median split
   over windows where it is defined.

   This matches the convention used in step 5.

2. Only the forward direction is used, phenotype -> compartment. Reverse
   columns are carried through so the other direction can be checked.

3. Burden is defined as a cell-count share rather than a window count,
   so a slide with many sparse tumour windows is not scored the same as
   one with a few dense ones.
--------------------------------------------------------------------------

Maps carry the tissue boundary, segmented from the H&E thumbnail on HSV
saturation (stained tissue is coloured, glass is not).

Reads   step5_interaction_state/tables/<SLIDE>_interaction_state.csv
        athena_2/<SLIDE>_athena_features.csv
        breast/st/<SLIDE>.h5ad  (thumbnail)
Writes  step6_tme_engagement/tables/<SLIDE>_tme_engagement.csv
        step6_tme_engagement/figures/<SLIDE>_tme_engagement.png
        step6_tme_engagement/tme_engagement_summary.csv
        step6_tme_engagement/burden_summary.csv
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
BASE_DIR    = config.STATE_OUT
STEP5_DIR   = BASE_DIR / "step5_interaction_state" / "tables"
ATHENA2_DIR = config.ATHENA2_DIR
ST_DIR      = config.BREAST_ST_DIR

OUT_DIR = BASE_DIR / "step6_tme_engagement"
CSV_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
for d in (OUT_DIR, CSV_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SLIDES  = "ALL"
SPLIT_Q = 0.50       # median split, for infiltration
ENRICH_EPS = 0.0     # interaction: above this counts as above chance

DRAW_OUTLINE  = True    # segment and draw the tissue boundary
MIN_AREA_FRAC = 0.002   # drop specks below this fraction of the thumbnail
MIN_PTS       = 30      # drop contours with fewer vertices

PHENOTYPES   = {"inflammatory": "immune", "neoplastic": "cancer"}
COMPARTMENTS = {"connective": "connective", "epithelial": "epithelial"}

ENGAGE_COLORS = {
    "not_present":        "#2B2B33",
    "low_engagement":     "#4C6E8A",
    "tme_engaged":        "#E4A11B",
    "epithelium_engaged": "#3FA796",
    "both_engaged":       "#D9534F",
}
ENGAGE_ORDER = ["not_present", "low_engagement", "tme_engaged",
                "epithelium_engaged", "both_engaged"]


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

        mask = rgb_to_hsv(rgb)[..., 1] > threshold_otsu(rgb_to_hsv(rgb)[..., 1])
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


def flag_high(df, col, kind):
    """High flag for one measure.

    kind == "interaction": split at zero, the permutation-null expectation.
    kind == "infiltration": split at the within-slide median over defined
    values, since the ratio is bounded below by zero rather than centred.
    """
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool), np.nan
    v = df[col].to_numpy(float)

    if kind == "interaction":
        return (np.nan_to_num(v, nan=-np.inf) > ENRICH_EPS), float(ENRICH_EPS)

    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.zeros(len(df), dtype=bool), np.nan
    thr = float(np.quantile(v[ok], SPLIT_Q))
    return (np.nan_to_num(v, nan=-np.inf) > thr), thr


# ============================================================
# RUN
# ============================================================
if SLIDES == "ALL":
    files = sorted(glob.glob(str(STEP5_DIR / "*_interaction_state.csv")))
else:
    files = [str(STEP5_DIR / f"{s}_interaction_state.csv") for s in SLIDES]

if not files:
    raise SystemExit(f"No step 5 tables in {STEP5_DIR}. Run interaction_state.py first.")

engage_rows, burden_rows = [], []

for i, path in enumerate(files, 1):
    slide = os.path.basename(path).replace("_interaction_state.csv", "")
    feat_path = ATHENA2_DIR / f"{slide}_athena_features.csv"
    if not feat_path.exists():
        print(f"SKIP {slide} (no athena_2 features)")
        continue

    df = pd.read_csv(path)
    feats = pd.read_csv(feat_path)
    need = [c for c in feats.columns
            if (c.startswith("interaction_") or c.startswith("infiltration_"))
            and c not in df.columns]
    if need:
        df = df.merge(feats[["window_id"] + need], on="window_id", how="left")

    print(f"\n[{i}/{len(files)}] {slide}: {len(df)} windows", flush=True)

    erow = {"slide_id": slide,
            "technology": df["technology"].iloc[0] if "technology" in df.columns else "",
            "n_windows": len(df)}

    # ---- eight measures, flagged ----
    for pheno, short in PHENOTYPES.items():
        present = df[f"{pheno}_count"].fillna(0).to_numpy() > 0

        hi = {}
        for comp in COMPARTMENTS:
            for kind in ("interaction", "infiltration"):
                col = f"{kind}_{pheno}_to_{comp}"
                flag, thr = flag_high(df, col, kind)
                name = f"{short}_{kind}_{comp}_high"
                df[name] = flag & present
                hi[(kind, comp)] = df[name].to_numpy()
                erow[f"{name}_n"] = int(df[name].sum())

        # engagement label from the interaction flags
        conn = hi[("interaction", "connective")]
        epit = hi[("interaction", "epithelial")]
        lab = np.full(len(df), "low_engagement", dtype=object)
        lab[conn & ~epit] = "tme_engaged"
        lab[~conn & epit] = "epithelium_engaged"
        lab[conn & epit]  = "both_engaged"
        lab[~present]     = "not_present"
        df[f"{short}_engagement"] = lab

        counts = pd.Series(lab).value_counts()
        parts = "  ".join(f"{k}={int(counts.get(k,0))}" for k in ENGAGE_ORDER
                          if counts.get(k, 0))
        print(f"   {short:<7} {parts}")
        for k in ENGAGE_ORDER:
            erow[f"{short}_{k}"] = int(counts.get(k, 0))

    engage_rows.append(erow)

    # ---- burden ----
    cell_cols = {p: f"{p}_count" for p in
                 ["inflammatory", "neoplastic", "connective", "epithelial", "dead"]}
    have = {p: c for p, c in cell_cols.items() if c in df.columns}
    total_cells = sum(df[c].fillna(0).sum() for c in have.values())

    brow = {"slide_id": slide,
            "technology": erow["technology"],
            "n_windows": len(df),
            "total_cells": int(total_cells)}
    brow["tumour_burden"] = (float(df["neoplastic_count"].fillna(0).sum() / total_cells)
                             if total_cells else np.nan)
    brow["immune_burden"] = (float(df["inflammatory_count"].fillna(0).sum() / total_cells)
                             if total_cells else np.nan)
    if "neoplastic_fraction" in df.columns:
        brow["tumour_burden_window_mean"] = float(df["neoplastic_fraction"].mean())
    if "inflammatory_fraction" in df.columns:
        brow["immune_burden_window_mean"] = float(df["inflammatory_fraction"].mean())

    both = df["window_class"] == 4
    brow["n_both"] = int(both.sum())
    if "interaction_state" in df.columns and both.sum():
        brow["niche_fraction"] = float(df.loc[both, "is_niche"].mean())
    else:
        brow["niche_fraction"] = np.nan
    burden_rows.append(brow)

    print(f"   burden  tumour={brow['tumour_burden']:.4f}  "
          f"immune={brow['immune_burden']:.4f}  "
          f"niche={brow['niche_fraction'] if not np.isnan(brow['niche_fraction']) else float('nan'):.3f}")

    df.to_csv(CSV_DIR / f"{slide}_tme_engagement.csv", index=False)

    outline = tissue_outline(slide)

    # ---- figure: engagement maps ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, (pheno, short) in zip(axes, PHENOTYPES.items()):
        ax.set_facecolor("#101018")
        rects, colours = [], []
        for r in df.itertuples(index=False):
            colours.append(ENGAGE_COLORS.get(getattr(r, f"{short}_engagement"), "#FFF"))
            rects.append(patches.Rectangle((r.xmin, r.ymin),
                                           r.xmax - r.xmin, r.ymax - r.ymin))
        ax.add_collection(PatchCollection(rects, facecolors=colours,
                                          edgecolors="none", linewidths=0))
        for seg in outline:
            ax.plot(seg[:, 0], seg[:, 1], color="white", linewidth=1.0, zorder=5)

        pad = 0.05 * (df["xmax"].max() - df["xmin"].min())
        ax.set_xlim(df["xmin"].min() - pad, df["xmax"].max() + pad)
        ax.set_ylim(df["ymin"].min() - pad, df["ymax"].max() + pad)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xlabel("X coordinate"); ax.set_ylabel("Y coordinate")
        ax.set_title(f"{short}: connective / epithelial engagement")

        counts = pd.Series(df[f"{short}_engagement"]).value_counts()
        handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=9,
                          markerfacecolor=ENGAGE_COLORS[k], markeredgecolor="none",
                          label=f"{k} (n={int(counts.get(k,0))})")
                   for k in ENGAGE_ORDER if counts.get(k, 0)]
        ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=True)

    fig.suptitle(f"{slide}: engagement with connective and epithelial compartments")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{slide}_tme_engagement.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================
e = pd.DataFrame(engage_rows); e.to_csv(OUT_DIR / "tme_engagement_summary.csv", index=False)
b = pd.DataFrame(burden_rows); b.to_csv(OUT_DIR / "burden_summary.csv", index=False)

print("\n" + "=" * 60)
print("COHORT TOTALS")
print("=" * 60)
total = int(e["n_windows"].sum())
print(f"slides: {len(e)}   windows: {total:,}")
print("interaction split at zero (permutation null); infiltration at the "
      "within-slide median")
for short in PHENOTYPES.values():
    print(f"\n  {short} engagement")
    for k in ENGAGE_ORDER:
        col = f"{short}_{k}"
        if col in e.columns:
            n = int(e[col].sum())
            print(f"    {k:<20} {n:>7,}  ({100*n/total:5.1f}%)")

print("\n  burden across slides")
for c in ["tumour_burden", "immune_burden", "niche_fraction"]:
    if c in b.columns:
        print(f"    {c:<16} median {b[c].median():.4f}   "
              f"range {b[c].min():.4f} to {b[c].max():.4f}")

print("\ndone ->", OUT_DIR)
