"""
Steps 3-4: spatial state per window, per phenotype. STRICT.

Every condition of a state must hold. Windows meeting none are reported as
"other" rather than assigned a best match.

  absent      the phenotype is not present in the window

  isolated    low abundance AND low self-interaction AND no clustering
              (the 25-50 um condition below does not hold, and L - r is
              not positive at any radius >= 100). Also windows holding fewer cells than Ripley's
              L requires, which cannot be clustered in any meaningful sense;
              isolated_reason records which route.

  dispersed   moderate abundance
              AND heterotypic interaction above its within-slide median
              (HETERO_RULE)
              AND low self-interaction
              AND L - r positive at 25-50 um

  aggregate   moderate or high abundance
              AND high self-interaction
              AND L - r positive at 25-50 um, peaking at a short radius

  broad       moderate or high abundance
              AND high self-interaction
              AND L - r positive at 25-50 um, peaking at a radius >= 100

  other       satisfies no state completely

--------------------------------------------------------------------------
WHY THE THRESHOLDS DIFFER BY STATISTIC

Ripley's L is transformed so that L(r) - r = 0 is the expected value under
complete spatial randomness. The sign is therefore interpretable without
choosing anything: positive means more neighbors within r than random
placement would give. States are separated by WHICH radii are positive.

Magnitude is deliberately not used. L - r is centered on zero under CSR at any
sample size, but its sampling spread shrinks as n grows, so a
window holding 60 cells produces larger values from noise alone than a
window holding 2,000 produces from real clustering. Cell counts here run
from 50 to 2,868. Thresholding on magnitude would partly sort windows by
density rather than by arrangement. Testing magnitude properly needs Monte
Carlo envelopes per window, which the ATHENA Ripley output does not
provide.

Magnitude IS compared within a window, across radii, to separate aggregate
from broad. The sample-size effect is common to every radius of the same
window, so the SHAPE of one window's curve is interpretable even though its
height cannot be compared to another window's. A tight aggregate peaks at a
short radius; an extended region peaks further out.

Self-interaction is enrichment against a permutation null and is positive in
nearly every window, so its sign does not discriminate. It is thresholded
within the slide at the median: high above, low below. A quartile split would
leave half the windows neither high nor low, so they would fail every state.

Abundance labels come from step 2.

Dispersed does not require an absence of clustering at radii >= 100: L - r is
cumulative, so short-range clustering nearly guarantees long-range positivity.
What separates dispersed from aggregate is the interaction type, heterotypic
versus high self-interaction, not the clustering pattern.
--------------------------------------------------------------------------

HETERO_RULE sets how dispersed reads "interaction is with other cell types":
mean heterotypic interaction enrichment above zero ("positive"), above its
within-slide median ("median"), or exceeding self-interaction ("exceeds").

Reads   step2_abundance/tables/<SLIDE>_abundance.csv
        athena_2/<SLIDE>_athena_features.csv
        breast/st/<SLIDE>.h5ad  (thumbnail)
Writes  step3_spatial_state/tables/<SLIDE>_spatial_state.csv
        step3_spatial_state/figures/<SLIDE>_spatial_state.png
        step3_spatial_state/spatial_state_summary.csv
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
STEP2_DIR = BASE_DIR / "step2_abundance" / "tables"
ATHENA2_DIR = config.ATHENA2_DIR
ST_DIR = config.BREAST_ST_DIR

OUT_DIR = BASE_DIR / "step3_spatial_state"
CSV_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
for d in (OUT_DIR, CSV_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SLIDES = "ALL"

SHORT_RADII = [25, 50]  # "short"
BROAD_RADII = [100, 150, 200]  # "broad"

SHORT_RULE = "all"  # "all": every short radius positive; "any": at least one
HETERO_RULE = "median"  # "positive": heterotypic > 0
# "median":   heterotypic above its within-slide median
# "exceeds":  heterotypic > self-interaction

SELF_SPLIT_Q = 0.50  # self-interaction: high above this quantile in slide,
# low below it. A median split so that high and low
# partition rather than leaving a middle band that
# satisfies no state.
RIPLEY_EPS = 0.0  # L - r must exceed this to count as positive

PHENOTYPES = {"inflammatory": "immune", "neoplastic": "cancer"}
ALL_TYPES = ["inflammatory", "neoplastic", "connective", "epithelial"]

DRAW_OUTLINE = True
MIN_AREA_FRAC = 0.002
MIN_PTS = 30

STATE_COLORS = {
    "absent": "#2B2B33",
    "isolated": "#4C6E8A",
    "dispersed": "#3FA796",
    "aggregate": "#E4A11B",
    "broad": "#D9534F",
    "other": "#9A9AA5",
}
ORDER = ["absent", "isolated", "dispersed", "aggregate", "broad", "other"]


def tissue_outline(slide):
    """Tissue contours in full-resolution slide coordinates, or []."""
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
        return [np.column_stack([c[:, 1] / scale, c[:, 0] / scale]) for c in cs]
    except Exception as ex:
        print(f"   outline unavailable ({type(ex).__name__})")
        return []


def safe_quantile(v, q):
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if len(v) else np.nan


def assign_states(df, pheno, short):
    n = len(df)
    abundance = df[f"{short}_abundance"].to_numpy()

    # ---- self-interaction, within-slide median split ----
    self_col = f"interaction_{pheno}_to_{pheno}"
    self_int = (
        df[self_col].to_numpy(float) if self_col in df.columns else np.full(n, np.nan)
    )
    thr_self = safe_quantile(self_int, SELF_SPLIT_Q)
    hi_self = np.nan_to_num(self_int, nan=-np.inf) > thr_self
    lo_self = ~hi_self

    # ---- heterotypic interaction ----
    het_cols = [
        c
        for c in (f"interaction_{pheno}_to_{o}" for o in ALL_TYPES if o != pheno)
        if c in df.columns
    ]
    het_int = (
        df[het_cols].to_numpy(float).mean(axis=1) if het_cols else np.full(n, np.nan)
    )
    if HETERO_RULE == "median":
        thr_het = safe_quantile(het_int, 0.50)
        het_ok = np.nan_to_num(het_int, nan=-np.inf) > thr_het
    elif HETERO_RULE == "exceeds":
        het_ok = np.nan_to_num(het_int, nan=-np.inf) > np.nan_to_num(
            self_int, nan=np.inf
        )
    else:
        het_ok = np.nan_to_num(het_int, nan=-np.inf) > 0.0

    # ---- Ripley: sign at each radius ----
    def pos(radii):
        cols = [f"ripley_l_minus_r_{pheno}_{r}um" for r in radii]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return np.zeros((n, 0), dtype=bool)
        arr = df[cols].to_numpy(float)
        return arr > RIPLEY_EPS

    short_pos = pos(SHORT_RADII)
    broad_pos = pos(BROAD_RADII)

    # full curve, for peak location within the window
    all_radii = SHORT_RADII + BROAD_RADII
    curve_cols = [f"ripley_l_minus_r_{pheno}_{r}um" for r in all_radii]
    have = [c in df.columns for c in curve_cols]
    curve = (
        np.column_stack(
            [
                df[c].to_numpy(float) if c in df.columns else np.full(n, np.nan)
                for c in curve_cols
            ]
        )
        if any(have)
        else np.full((n, len(all_radii)), np.nan)
    )
    ok_row = np.isfinite(curve).any(axis=1)
    peak_idx = np.full(n, -1, dtype=int)
    if ok_row.any():
        peak_idx[ok_row] = np.nanargmax(curve[ok_row], axis=1)
    radii_arr = np.array(all_radii, dtype=float)
    peak_radius = np.where(
        peak_idx >= 0, radii_arr[np.clip(peak_idx, 0, len(all_radii) - 1)], np.nan
    )
    peak_short = np.isin(peak_radius, SHORT_RADII)

    ripley_missing = np.zeros(n, dtype=bool)
    all_cols = [f"ripley_l_minus_r_{pheno}_{r}um" for r in SHORT_RADII + BROAD_RADII]
    all_cols = [c for c in all_cols if c in df.columns]
    if all_cols:
        ripley_missing = ~np.isfinite(df[all_cols].to_numpy(float)).any(axis=1)

    if short_pos.shape[1]:
        short_ok = (
            short_pos.all(axis=1) if SHORT_RULE == "all" else short_pos.any(axis=1)
        )
    else:
        short_ok = np.zeros(n, dtype=bool)

    if broad_pos.shape[1]:
        n_broad_pos = broad_pos.sum(axis=1)
    else:
        n_broad_pos = np.zeros(n, dtype=int)

    any_pos = short_ok | (n_broad_pos > 0)

    # ---- abundance ----
    is_absent = abundance == "absent"
    low_ab = abundance == "low"
    mod_ab = abundance == "moderate"
    modhigh = np.isin(abundance, ["moderate", "high"])

    # ---- strict conjunctions ----
    state = np.full(n, "other", dtype=object)

    reason = np.full(n, "", dtype=object)

    # most specific first
    state[modhigh & hi_self & short_ok & ~peak_short] = "broad"
    state[modhigh & hi_self & short_ok & peak_short] = "aggregate"
    state[mod_ab & het_ok & short_ok & ~hi_self] = "dispersed"
    state[low_ab & lo_self & ~any_pos] = "isolated"
    reason[state == "isolated"] = "measured_unclustered"

    # below Ripley's cell minimum: no clustering statistic exists, and a
    # 500 um window holding fewer cells than Ripley's minimum is sparse
    too_few = ripley_missing & ~is_absent
    state[too_few] = "isolated"
    reason[too_few] = "below_ripley_min_cells"

    state[is_absent] = "absent"
    reason[is_absent] = ""

    diag = {
        f"{short}_self_interaction": self_int,
        f"{short}_het_interaction": het_int,
        f"{short}_self_high": hi_self,
        f"{short}_self_low": lo_self,
        f"{short}_short_positive": short_ok,
        f"{short}_n_broad_positive": n_broad_pos,
        f"{short}_ripley_missing": ripley_missing,
        f"{short}_ripley_peak_radius": peak_radius,
        f"{short}_isolated_reason": reason,
    }
    thr = {f"{short}_self_median": thr_self}
    return state, diag, thr


# ============================================================
# RUN
# ============================================================
if SLIDES == "ALL":
    files = sorted(glob.glob(str(STEP2_DIR / "*_abundance.csv")))
else:
    files = [str(STEP2_DIR / f"{s}_abundance.csv") for s in SLIDES]

if not files:
    raise SystemExit(
        f"No step 2 tables in {STEP2_DIR}. Run measure_abundance.py first."
    )

summary_rows = []

for i, path in enumerate(files, 1):
    slide = os.path.basename(path).replace("_abundance.csv", "")
    feat_path = ATHENA2_DIR / f"{slide}_athena_features.csv"
    if not feat_path.exists():
        print(f"SKIP {slide} (no athena_2 features)")
        continue

    df = pd.read_csv(path)
    feats = pd.read_csv(feat_path)
    wanted = [
        c
        for c in feats.columns
        if c == "window_id"
        or c.startswith("interaction_")
        or c.startswith("ripley_l_minus_r_")
    ]
    df = df.merge(feats[wanted], on="window_id", how="left")

    print(f"\n[{i}/{len(files)}] {slide}: {len(df)} windows", flush=True)

    row = {
        "slide_id": slide,
        "technology": df["technology"].iloc[0] if "technology" in df.columns else "",
        "n_windows": len(df),
    }

    for pheno, short in PHENOTYPES.items():
        state, diag, thr = assign_states(df, pheno, short)
        df[f"{short}_spatial_state"] = state
        for k, v in diag.items():
            df[k] = v

        counts = pd.Series(state).value_counts()
        parts = "  ".join(
            f"{lab}={int(counts.get(lab, 0))}" for lab in ORDER if counts.get(lab, 0)
        )
        print(f"   {short:<7} {parts}")
        for lab in ORDER:
            row[f"{short}_{lab}"] = int(counts.get(lab, 0))
        row.update(thr)

    summary_rows.append(row)
    df.to_csv(CSV_DIR / f"{slide}_spatial_state.csv", index=False)

    outline = tissue_outline(slide)

    # ---- two-panel map ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, (pheno, short) in zip(axes, PHENOTYPES.items()):
        ax.set_facecolor("#101018")
        rects, colors = [], []
        for r in df.itertuples(index=False):
            colors.append(
                STATE_COLORS.get(getattr(r, f"{short}_spatial_state"), "#FFF")
            )
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
        ax.set_title(f"{short} spatial state")

        counts = pd.Series(df[f"{short}_spatial_state"]).value_counts()
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=9,
                markerfacecolor=STATE_COLORS[lab],
                markeredgecolor="none",
                label=f"{lab} (n={int(counts.get(lab, 0))})",
            )
            for lab in ORDER
            if counts.get(lab, 0)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=True)

    fig.suptitle(f"{slide}: spatial state by window (strict)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{slide}_spatial_state.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================
s = pd.DataFrame(summary_rows)
s.to_csv(OUT_DIR / "spatial_state_summary.csv", index=False)

print("\n" + "=" * 60)
print("COHORT TOTALS  (strict: all conditions must hold)")
print("=" * 60)
total = int(s["n_windows"].sum())
print(f"slides: {len(s)}   windows: {total:,}")
print(
    f"Ripley by sign of L - r; short {SHORT_RADII} um ({SHORT_RULE}), "
    f"broad {BROAD_RADII} um"
)
print(
    f"self-interaction: median split within slide;  "
    f"heterotypic rule: {HETERO_RULE};  "
    f"aggregate vs broad by peak radius within the window"
)
for short in PHENOTYPES.values():
    print(f"\n  {short}")
    for lab in ORDER:
        col = f"{short}_{lab}"
        if col in s.columns:
            n = int(s[col].sum())
            print(f"    {lab:<10} {n:>7,}  ({100*n/total:5.1f}%)")

print("\ndone ->", OUT_DIR)
