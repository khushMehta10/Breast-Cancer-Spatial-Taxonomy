"""
Interaction-state maps -- one tile per window over the H&E thumbnail,
colored by Step 5 interaction state (Distant / Bordering / Infiltrated).

Input : 3_spatial_state/output/step5_interaction_state/tables/<SLIDE>_interaction_state.csv
Image : breast/st/<SLIDE>.h5ad  (downscaled_fullres thumbnail + scalefactor)
Output: 3_spatial_state/output/step5_interaction_state/state_maps/<SLIDE>_interaction_state.png
"""

import traceback
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

# ============================================================
# CONFIG
# ============================================================
BATCH_ALL = True  # True = every table in TABLE_DIR
SLIDES = ["SPA105"]  # used only when BATCH_ALL = False

ST_DIR = config.BREAST_ST_DIR
TABLE_DIR = config.STEP5_TABLES
FIG_DIR = config.STATE_OUT / "step5_interaction_state" / "state_maps"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.72  # tile transparency so the H&E still shows through
GRID_EDGE = "white"
GRID_LW = 0.3
HATCH_EMPTY = True  # True -> zero-cell windows hatched
SHOW_NICHE = True  # True -> niche windows drawn in their own color

# interaction state -> color (state colors match interaction_state.py; the two
# background classes are lighter so the H&E shows through)
STATE_COLORS = {
    "not_applicable": "#E2E8EB",  # one compartment absent   pale gray
    "unlabeled": "#B9C0C6",  # co-occurring, no state   gray
    "distant_co_occurrence": "#4C6E8A",  # steel blue
    "bordering": "#E4A11B",  # amber
    "infiltrated": "#35B779",  # green
    "niche": "#D9534F",  # red, infiltrated + niche
}
STATE_LABELS = {
    "not_applicable": "One compartment absent",
    "unlabeled": "Co-occurring, unlabeled",
    "distant_co_occurrence": "Distant co-occurrence",
    "bordering": "Bordering",
    "infiltrated": "Infiltrated",
    "niche": "Niche (infiltrated)",
}
ORDER = [
    "not_applicable",
    "unlabeled",
    "distant_co_occurrence",
    "bordering",
    "infiltrated",
    "niche",
]
NAMED = ["distant_co_occurrence", "bordering", "infiltrated", "niche"]
COUNT_COLS = [
    "inflammatory_count",
    "neoplastic_count",
    "connective_count",
    "epithelial_count",
    "dead_count",
]


def find_h5ad(slide_id):
    p = ST_DIR / f"{slide_id}.h5ad"
    return p if p.exists() else None


def render_slide(slide_id):
    table = TABLE_DIR / f"{slide_id}_interaction_state.csv"
    if not table.exists():
        print(f"SKIP {slide_id}: no table")
        return

    df = pd.read_csv(table).dropna(subset=["xmin", "xmax", "ymin", "ymax"])
    if df.empty:
        print(f"SKIP {slide_id}: no windows with geometry")
        return

    h5 = find_h5ad(slide_id)
    if h5 is None:
        print(f"SKIP {slide_id}: no h5ad in {ST_DIR}")
        return

    have = [c for c in COUNT_COLS if c in df.columns]
    total = df[have].fillna(0).sum(axis=1).to_numpy()

    # display state: niche promoted out of infiltrated when SHOW_NICHE
    display = df["interaction_state"].astype(str).to_numpy()
    if SHOW_NICHE and "is_niche" in df.columns:
        nm = (
            df["is_niche"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "1.0", "yes"])
            .to_numpy()
        )
        display = np.where(nm, "niche", display)

    adata = ad.read_h5ad(h5)
    lib = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]
    img = np.asarray(lib["images"]["downscaled_fullres"])
    scale = float(lib["scalefactors"]["tissue_downscaled_fullres_scalef"])
    H, W = img.shape[0], img.shape[1]
    extent = [0, W / scale, H / scale, 0]  # full-resolution frame, y down

    box_w = float(np.median(df["xmax"] - df["xmin"]))
    box_h = float(np.median(df["ymax"] - df["ymin"]))

    pad0 = max(box_w, box_h) * 0.10
    dx = (df["xmax"].max() - df["xmin"].min()) + 2 * pad0
    dy = (df["ymax"].max() - df["ymin"].min()) + 2 * pad0
    aspect = float(dx) / float(dy) if dy else 1.0
    BASE = 8.0
    figsize = (BASE, BASE / aspect) if aspect >= 1 else (BASE * aspect, BASE)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"{slide_id}\nTumor-immune interaction state")
    ax.imshow(img, extent=extent, interpolation="nearest", zorder=0)

    n_hatch = 0
    by_state = {k: 0 for k in STATE_COLORS}
    for row, tot, st in zip(df.itertuples(index=False), total, display):
        x0, y0 = row.xmin, row.ymin
        w, h = row.xmax - row.xmin, row.ymax - row.ymin
        if HATCH_EMPTY and tot <= 0:
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    w,
                    h,
                    facecolor=(0.5, 0.5, 0.5, 0.12),
                    hatch="////",
                    edgecolor=(0.35, 0.35, 0.35, 0.9),
                    linewidth=0,
                    zorder=2,
                )
            )
            n_hatch += 1
        else:
            named = st in ("distant_co_occurrence", "bordering", "infiltrated", "niche")
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0),
                    w,
                    h,
                    facecolor=STATE_COLORS.get(st, "#000000"),
                    alpha=ALPHA,
                    edgecolor=GRID_EDGE,
                    linewidth=GRID_LW,
                    zorder=3 if named else 2,
                )
            )
            by_state[st] = by_state.get(st, 0) + 1

    pad = max(box_w, box_h) * 0.10
    ax.set_xlim(df["xmin"].min() - pad, df["xmax"].max() + pad)
    ax.set_ylim(df["ymax"].max() + pad, df["ymin"].min() - pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Slide x-coordinate")
    ax.set_ylabel("Slide y-coordinate")

    handles = [
        patches.Patch(
            facecolor=STATE_COLORS[s],
            edgecolor="none",
            alpha=ALPHA,
            label=f"{STATE_LABELS[s]}  (n={by_state.get(s, 0)})",
        )
        for s in NAMED
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        fontsize=9,
        title="Interaction state",
        borderpad=0.6,
        labelspacing=0.45,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_alpha(0.92)
    leg.set_zorder(20)

    out = FIG_DIR / f"{slide_id}_interaction_state.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = ", ".join(
        f"{STATE_LABELS[s]}={by_state.get(s, 0)}"
        for s in ORDER[2:]
        if by_state.get(s, 0)
    )
    print(
        f"{slide_id}: {len(df)} windows ({summary or 'none named'}, "
        f"hatched={n_hatch}) -> {out.name}"
    )


# ============================================================
# RUN
# ============================================================
if BATCH_ALL:
    ids = sorted(
        p.name.replace("_interaction_state.csv", "")
        for p in TABLE_DIR.glob("*_interaction_state.csv")
    )
else:
    ids = SLIDES

print(f"rendering {len(ids)} slide(s) -> {FIG_DIR}", flush=True)
ok = fail = 0
for i, sid in enumerate(ids, 1):
    print(f"[{i}/{len(ids)}] {sid}", flush=True)
    try:
        render_slide(sid)
        ok += 1
    except Exception:
        fail += 1
        print(f"ERROR {sid}:")
        traceback.print_exc()
        plt.close("all")
print(f"\ndone: {ok} rendered, {fail} failed -> {FIG_DIR}")
