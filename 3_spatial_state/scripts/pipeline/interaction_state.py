"""
Step 5: tumour-immune spatial relation, for windows containing both.

DETECTION, not classification. A window carries a label only if it satisfies
every condition of that label. Windows satisfying none are "unlabeled". There
is no scoring, no best match and no tie-breaking: the three states are
mutually exclusive by construction, since distant requires interaction below
chance while bordering and infiltrated require it above, and bordering
requires high modularity while infiltrated requires low.

  distant_co_occurrence   interaction below chance in both directions
                          immune clustered (aggregate or broad)
                          tumour clustered (aggregate or broad)
                          modularity high

  bordering               interaction above chance in both directions
                          immune -> tumour infiltration low
                          immune clustered (aggregate or broad)
                          at least BORDER_MIN_TUMOR tumour nuclei
                          modularity high

  infiltrated             interaction above chance in both directions
                          infiltration high in both directions
                          modularity low

  is_niche                a refinement of infiltrated, not a separate state:
                          infiltrated, plus moderate-or-high abundance in both
                          compartments, plus high local richness

  dominance               a descriptor applied to every co-occurring window,
                          from phenotype fractions alone. Not a detection.

THRESHOLDS

  Interaction is an enrichment against a permutation null, so zero is the
  value expected by chance and its sign is meaningful: above chance means more
  cross-type contact than random placement would give.

  Infiltration is a ratio of cross-type to same-type edges, bounded below by
  zero rather than centred on it, so it takes a within-slide median split.
  Modularity, richness and the phenotype fractions do the same.

  Clustering and abundance come from steps 3 and 2 respectively. Both
  compartments must be clustered for distant_co_occurrence; bordering requires
  only the immune compartment to be clustered. Infiltrated does not use
  clustering, since infiltrated immune cells are dispersed among tumour rather
  than aggregated with each other.

Reads   step3_spatial_state/tables/<SLIDE>_spatial_state.csv
        athena_2/<SLIDE>_athena_features.csv
Writes  step5_interaction_state/tables/<SLIDE>_interaction_state.csv
        step5_interaction_state/figures/<SLIDE>_interaction_state.png
        step5_interaction_state/interaction_state_summary.csv
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
import glob, os, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

# ============================================================
# CONFIG
# ============================================================
BASE_DIR    = config.STATE_OUT
STEP3_DIR   = BASE_DIR / "step3_spatial_state" / "tables"
ATHENA2_DIR = config.ATHENA2_DIR

OUT_DIR = BASE_DIR / "step5_interaction_state"
CSV_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
for d in (OUT_DIR, CSV_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SLIDES  = "ALL"
SPLIT_Q = 0.50          # within-slide split for infiltration, modularity,
                        # richness and fractions
ENRICH_EPS = 0.0        # interaction: above this counts as above chance

CLUSTERED_STATES = ("aggregate", "broad")
ABUNDANT_TIERS   = ("moderate", "high")

# Bordering requires the tumor compartment to be a group rather than a scatter.
# Five is where tumour self-interaction flips from 22% of windows above chance
# (3-4 nuclei) to 84% (5-9 nuclei), so the threshold is set by the data.
BORDER_MIN_TUMOR = 5
TUMOR_COUNT_COL  = "neoplastic_count"

# source columns
COLS = {
    "int_i2c":  "interaction_inflammatory_to_neoplastic",
    "int_c2i":  "interaction_neoplastic_to_inflammatory",
    "inf_i2c":  "infiltration_inflammatory_to_neoplastic",
    "inf_c2i":  "infiltration_neoplastic_to_inflammatory",
    "modular":  "global_modularity",
    "richness": "local_richness_mean",
    "frac_i":   "inflammatory_fraction",
    "frac_c":   "neoplastic_fraction",
}
SIGN_METRICS = {"int_i2c", "int_c2i"}     # thresholded at zero
CAT_COLS = {
    "sp_i": "immune_spatial_state",
    "sp_c": "cancer_spatial_state",
    "ab_i": "immune_abundance",
    "ab_c": "cancer_abundance",
}

STATE_COLORS = {
    "not_applicable":        "#2B2B33",
    "unlabeled":             "#4A4A55",
    "distant_co_occurrence": "#4C6E8A",
    "bordering":             "#E4A11B",
    "infiltrated":           "#35B779",
    "niche":                 "#D9534F",   # display only: infiltrated + niche
}
ORDER = ["not_applicable", "unlabeled", "distant_co_occurrence",
         "bordering", "infiltrated"]
DOM_ORDER = ["tumour_dominant", "immune_dominant", "balanced"]


# ============================================================
# FLAGS
# ============================================================
def hi_lo(df, mask):
    """High/low booleans per metric, thresholded over co-occurring windows."""
    n = len(df)
    hi, lo = {}, {}
    for key, col in COLS.items():
        v = df[col].to_numpy(float)
        if key in SIGN_METRICS:
            hi[key] = np.nan_to_num(v, nan=-np.inf) > ENRICH_EPS
            lo[key] = np.nan_to_num(v, nan=np.inf) < ENRICH_EPS
            continue
        ref = v[mask & np.isfinite(v)]
        if len(ref) == 0:
            hi[key] = np.zeros(n, dtype=bool)
            lo[key] = np.zeros(n, dtype=bool)
            continue
        thr = float(np.quantile(ref, SPLIT_Q))
        hi[key] = np.nan_to_num(v, nan=-np.inf) > thr
        # An undefined value is neither high nor low, so a window never
        # satisfies a condition that could not be evaluated.
        lo[key] = (~hi[key]) & np.isfinite(v)
    return hi, lo


def assign(df):
    """Strict labels, a niche flag and a dominance descriptor."""
    n = len(df)
    both = (df["window_class"].to_numpy() == 4)

    state = np.full(n, "not_applicable", dtype=object)
    dom   = np.full(n, "not_applicable", dtype=object)
    niche = np.zeros(n, dtype=bool)
    diag  = {}

    if both.sum() == 0:
        return state, niche, dom, diag, both

    hi, lo = hi_lo(df, both)

    clustered_i = np.isin(df[CAT_COLS["sp_i"]].to_numpy(object), CLUSTERED_STATES)
    clustered_c = np.isin(df[CAT_COLS["sp_c"]].to_numpy(object), CLUSTERED_STATES)
    abundant_i  = np.isin(df[CAT_COLS["ab_i"]].to_numpy(object), ABUNDANT_TIERS)
    abundant_c  = np.isin(df[CAT_COLS["ab_c"]].to_numpy(object), ABUNDANT_TIERS)

    contact_lo = lo["int_i2c"] & lo["int_c2i"]
    contact_hi = hi["int_i2c"] & hi["int_c2i"]
    infil_hi   = hi["inf_i2c"] & hi["inf_c2i"]

    q_distant = both & contact_lo & clustered_i & clustered_c & hi["modular"]
    # Bordering tests immune->tumor infiltration only. The tumor->immune
    # direction is undefined in 86% of candidate windows, because a sparse
    # tumor compartment forms no tumor-to-tumor edges and the ratio has a zero denominator.
    # Requiring an organized immune body in its place keeps the meaning of
    # the state, contact without penetration, without ever naming a window
    # on the basis of a value that was not measured.
    if TUMOR_COUNT_COL in df.columns:
        tumor_group = df[TUMOR_COUNT_COL].to_numpy(float) >= BORDER_MIN_TUMOR
    else:
        print(f"   ERROR: {TUMOR_COUNT_COL} not found; bordering minimum tumor "
              f"count NOT applied.", file=sys.stderr)
        tumor_group = np.ones(n, dtype=bool)
    q_border  = (both & contact_hi & lo["inf_i2c"] & clustered_i
                 & hi["modular"] & tumor_group)
    q_infil   = both & contact_hi & infil_hi & lo["modular"]

    # mutual exclusivity holds by construction; checked here
    overlap = (q_distant.astype(int) + q_border.astype(int) +
               q_infil.astype(int)) > 1
    if overlap.any():
        print(f"   ERROR: {int(overlap.sum())} windows qualify for more than "
              f"one state. Definitions are not exclusive.", file=sys.stderr)

    state[both]      = "unlabeled"
    state[q_distant] = "distant_co_occurrence"
    state[q_border]  = "bordering"
    state[q_infil]   = "infiltrated"

    niche = q_infil & abundant_i & abundant_c & hi["richness"]

    dom[both] = "balanced"
    dom[both & hi["frac_c"] & lo["frac_i"]] = "tumour_dominant"
    dom[both & hi["frac_i"] & lo["frac_c"]] = "immune_dominant"

    diag = {
        "int_i2c_above_chance": hi["int_i2c"],
        "int_c2i_above_chance": hi["int_c2i"],
        "infiltration_i2c_high": hi["inf_i2c"],
        "infiltration_c2i_high": hi["inf_c2i"],
        "modularity_high":       hi["modular"],
        "richness_high":         hi["richness"],
        "immune_clustered":      clustered_i,
        "cancer_clustered":      clustered_c,
        "immune_abundant":       abundant_i,
        "cancer_abundant":       abundant_c,
        "qualifies_distant":     q_distant,
        "qualifies_bordering":   q_border,
        "qualifies_infiltrated": q_infil,
    }
    return state, niche, dom, diag, both


# ============================================================
# RUN
# ============================================================
if SLIDES == "ALL":
    files = sorted(glob.glob(str(STEP3_DIR / "*_spatial_state.csv")))
else:
    files = [str(STEP3_DIR / f"{s}_spatial_state.csv") for s in SLIDES]

if not files:
    raise SystemExit(f"No step 3 tables in {STEP3_DIR}. Run spatial_state.py first.")

summary_rows, warned = [], False

for i, path in enumerate(files, 1):
    slide = os.path.basename(path).replace("_spatial_state.csv", "")
    feat_path = ATHENA2_DIR / f"{slide}_athena_features.csv"
    if not feat_path.exists():
        print(f"SKIP {slide} (no athena_2 features)")
        continue

    df = pd.read_csv(path)
    feats = pd.read_csv(feat_path)
    need = [c for c in COLS.values() if c in feats.columns and c not in df.columns]
    if need:
        df = df.merge(feats[["window_id"] + need], on="window_id", how="left")

    missing = ([c for c in COLS.values() if c not in df.columns] +
               [c for c in CAT_COLS.values() if c not in df.columns])
    if missing:
        raise SystemExit(
            f"{slide}: required columns missing, refusing to run:\n   " +
            "\n   ".join(missing))

    state, niche, dom, diag, both = assign(df)
    df["interaction_state"] = state
    df["is_niche"] = niche
    df["dominance"] = dom
    for k, v in diag.items():
        df[k] = v

    counts = pd.Series(state).value_counts()
    n_both = int(both.sum())
    n_lab  = sum(int(counts.get(k, 0)) for k in
                 ("distant_co_occurrence", "bordering", "infiltrated"))
    parts = "  ".join(f"{k}={int(counts.get(k, 0))}" for k in ORDER[2:]
                      if counts.get(k, 0))
    print(f"\n[{i}/{len(files)}] {slide}: {len(df)} windows, {n_both} with both",
          flush=True)
    if n_both:
        print(f"   {parts}   niche={int(niche.sum())}   "
              f"unlabeled={int(counts.get('unlabeled', 0))}")

    row = {"slide_id": slide,
           "technology": df["technology"].iloc[0] if "technology" in df.columns else "",
           "n_windows": len(df), "n_both": n_both,
           "n_labeled": n_lab, "n_niche": int(niche.sum())}
    for k in ORDER:
        row[k] = int(counts.get(k, 0))
    for d in DOM_ORDER:
        row[f"dom_{d}"] = int((dom == d).sum())
    summary_rows.append(row)

    df.to_csv(CSV_DIR / f"{slide}_interaction_state.csv", index=False)

    # ---- map: niche drawn distinctly, though it is a subset of infiltrated ----
    display = state.copy()
    display[niche] = "niche"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#101018")
    rects, colours = [], []
    for r, lab in zip(df.itertuples(index=False), display):
        colours.append(STATE_COLORS.get(lab, "#FFFFFF"))
        rects.append(patches.Rectangle((r.xmin, r.ymin),
                                       r.xmax - r.xmin, r.ymax - r.ymin))
    ax.add_collection(PatchCollection(rects, facecolors=colours,
                                      edgecolors="none", linewidths=0))

    pad = 0.05 * (df["xmax"].max() - df["xmin"].min())
    ax.set_xlim(df["xmin"].min() - pad, df["xmax"].max() + pad)
    ax.set_ylim(df["ymin"].min() - pad, df["ymax"].max() + pad)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"{slide}: tumour-immune spatial relation")

    shown = pd.Series(display).value_counts()
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=9,
                      markerfacecolor=STATE_COLORS[lab], markeredgecolor="none",
                      label=f"{lab} (n={int(shown.get(lab, 0))})")
               for lab in ORDER + ["niche"] if shown.get(lab, 0)]
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{slide}_interaction_state.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================
s = pd.DataFrame(summary_rows)
s.to_csv(OUT_DIR / "interaction_state_summary.csv", index=False)

total = int(s["n_windows"].sum())
both  = int(s["n_both"].sum())
print("\n" + "=" * 64)
print("COHORT TOTALS   (strict: every condition must hold)")
print("=" * 64)
print(f"slides: {len(s)}   windows: {total:,}   with both phenotypes: {both:,}")
print("interaction split at zero (permutation null); infiltration, modularity,")
print("richness and fractions by median within slide; clustering and abundance")
print("from steps 3 and 2")
print()
for lab in ORDER:
    n = int(s[lab].sum())
    denom = total if lab == "not_applicable" else both
    scope = "of all" if lab == "not_applicable" else "of both"
    pct = 100 * n / denom if denom else 0
    print(f"   {lab:<24} {n:>7,}  ({pct:5.1f}% {scope})")
n_niche = int(s["n_niche"].sum())
print(f"   {'  of which niche':<24} {n_niche:>7,}  ({100*n_niche/both:5.1f}% of both)")
print()
print("   dominance descriptor, over co-occurring windows")
for d in DOM_ORDER:
    n = int(s[f"dom_{d}"].sum())
    print(f"   {d:<24} {n:>7,}  ({100*n/both:5.1f}%)")

print("\ndone ->", OUT_DIR)
