"""Overview figure of the pipeline.

Stages: WSI -> 500 um windows -> CellViT (five phenotypes, immune and tumor highlighted) ->
ATHENA features -> spatial state definition.

The ATHENA panel shows the six quantities named in Methods 3.2, each labelled
with the symbol used in the text: abundance, self-interaction, cross-population
interaction, directed infiltration, graph modularity, and the Ripley
clustering statistic evaluated at five radii.

Immune is Okabe-Ito bluish green and tumor vermilion, matching the H&E
overlay figure.

  usage:  python 3_spatial_state/scripts/figures/fig_pipeline_overview.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Ellipse, FancyArrow

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

SLIDE = "SPA106"
ST_DIR = config.BREAST_ST_DIR
STEP5 = config.STEP5_TABLES
OUT_DIR = config.FIGURES_OUT

IMM, TUM, OTH = "#009E73", "#D55E00", "#B9C0C7"
INK, GREY, LINE, PALE = "#111111", "#666666", "#2B2B2B", "#B8BEC4"
rng = np.random.default_rng(11)

W, H = 176.0, 58.0
TOP, BOT = 50.0, 5.0
fig = plt.figure(figsize=(6.5, 6.5 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off"); ax.set_aspect("equal")

STAGE = [("WSI", 2.0, 22.0),
         ("500 $\\mu$m windows", 30.0, 22.0),
         ("CellViT", 58.0, 24.0),
         ("ATHENA features", 88.0, 42.0),
         ("Spatial state definition", 136.0, 38.0)]


def frame(x, w, title):
    ax.add_patch(FancyBboxPatch((x, BOT), w, TOP - BOT,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec=LINE, lw=1.6, zorder=1))
    ax.text(x + w / 2, TOP + 1.4, title, ha="center", va="bottom",
            fontsize=8.5, color=INK, zorder=5)


for name, x, w in STAGE:
    frame(x, w, name)
for i in range(len(STAGE) - 1):
    x0 = STAGE[i][1] + STAGE[i][2] + 1.0
    x1 = STAGE[i + 1][1] - 1.0
    ax.add_patch(FancyArrow(x0, (TOP + BOT) / 2, x1 - x0, 0, width=1.0,
                            head_width=2.8, head_length=2.2,
                            length_includes_head=True, fc=GREY, ec=GREY,
                            zorder=4))

# ------------------------------------------------------------- slide data
import anndata as ad
import pandas as pd

h5 = ST_DIR / f"{SLIDE}.h5ad"
tbl_path = STEP5 / f"{SLIDE}_interaction_state.csv"
if not (h5.exists() and tbl_path.exists()):
    sys.exit(f"{SLIDE}: h5ad or step 5 table not found")
adata = ad.read_h5ad(h5, backed="r")
lib = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]
thumb = np.asarray(lib["images"]["downscaled_fullres"])
SF = float(lib["scalefactors"]["tissue_downscaled_fullres_scalef"])
if thumb.dtype != np.uint8:
    thumb = (255 * thumb.astype(float) / max(1.0, thumb.max())).astype(np.uint8)
if thumb.ndim == 3 and thumb.shape[2] == 4:
    thumb = thumb[..., :3]
HE = thumb
wt = pd.read_csv(tbl_path)
TH, TW = HE.shape[0], HE.shape[1]
WINS = [((r.xmin * SF) / TW, (r.ymin * SF) / TH,
         ((r.xmax - r.xmin) * SF) / TW,
         ((r.ymax - r.ymin) * SF) / TH) for r in wt.itertuples()]
print(f"{SLIDE}: thumbnail {TW} x {TH}, {len(WINS)} windows")

# ---------------------------------------------------------------- 1. WSI
x, w = STAGE[0][1], STAGE[0][2]
bx, by = x + 2.0, BOT + 2.5
bw, bh = w - 4.0, TOP - BOT - 5.0
ih, iw = HE.shape[0], HE.shape[1]
sc = min(bw / iw, bh / ih)
dw, dh = iw * sc, ih * sc
ox, oy = bx + (bw - dw) / 2, by + (bh - dh) / 2
ax.imshow(HE, extent=(ox, ox + dw, oy, oy + dh), zorder=2,
          interpolation="antialiased")
ax.add_patch(Rectangle((ox, oy), dw, dh, fc="none", ec="#2B2B2B", lw=0.8,
                       zorder=3))

# ------------------------------------------------------ 2. tissue windows
x, w = STAGE[1][1], STAGE[1][2]
bx2 = x + 2.0
ox2 = bx2 + (bw - dw) / 2
ax.imshow(HE, extent=(ox2, ox2 + dw, oy, oy + dh), zorder=2, alpha=.55,
          interpolation="antialiased")
boxes = [(ox2 + fx * dw, oy + dh - (fy + fh) * dh, fw * dw, fh * dh)
         for fx, fy, fw, fh in WINS]
for bxx, byy, bww, bhh in boxes:
    ax.add_patch(Rectangle((bxx, byy), bww, bhh, fc="none", ec="#2B2B2B",
                           lw=0.35, zorder=4))
ax.add_patch(Rectangle((ox2, oy), dw, dh, fc="none", ec="#2B2B2B",
                       lw=0.8, zorder=3))

# ------------------------------------------------- 3. CellViT, five types
x, w = STAGE[2][1], STAGE[2][2]
cx = x + w / 2
ax.add_patch(Rectangle((x + 2.5, TOP - 9.0), w - 5.0, 5.6, fc="#F4F6F8",
                       ec="#2B2B2B", lw=1.0, zorder=2))
ax.text(cx, TOP - 6.2, "CellViT", ha="center", va="center", fontsize=7.5,
        color=INK, zorder=3)
ax.add_patch(FancyArrow(cx, TOP - 9.6, 0, -2.4, width=0.6, head_width=1.9,
                        head_length=1.5, length_includes_head=True,
                        fc=GREY, ec=GREY, zorder=3))

TYPES = [("immune", IMM, True), ("tumor", TUM, True),
         ("connective", OTH, False), ("epithelial", OTH, False),
         ("dead", OTH, False)]
y0 = TOP - 15.5
for k, (lab, col, keep) in enumerate(TYPES):
    yy = y0 - k * 6.0
    if keep and lab == "tumor":
        ax.add_patch(Rectangle((x + 5.2, yy - 1.2), 2.4, 2.4, fc=col, ec=col,
                               zorder=3))
    else:
        ax.add_patch(Ellipse((x + 6.4, yy), 2.8, 2.2,
                             fc=col if keep else "white", ec=col,
                             lw=1.3 if keep else 0.9, zorder=3))
    ax.text(x + 9.0, yy, lab, ha="left", va="center",
            fontsize=6.8 if keep else 6.2,
            color=INK if keep else "#8A8A8A", zorder=3)

# ------------------------------------------------- 4. ATHENA feature panels
x, w = STAGE[3][1], STAGE[3][2]
GX, GY = 1.6, 1.2
cw = (w - 2 * GX - 1.2) / 2
ch = (TOP - BOT - 2 * GY - 2 * 1.0) / 3
COLX = [x + GX, x + GX + cw + 1.2]
ROWY = [TOP - GY - (r + 1) * ch - r * 1.0 for r in range(3)]


def dot(px, py, col, sq=False, s=.85):
    if sq:
        ax.add_patch(Rectangle((px - s, py - s), 2 * s, 2 * s, fc=col, ec=col,
                               zorder=4))
    else:
        ax.add_patch(Circle((px, py), s, fc=col, ec=col, zorder=4))


def abundance(px, py, pw, ph):
    for _ in range(11):
        dot(px + pw * .40 + rng.normal(0, 2.0), py + ph * .64 + rng.normal(0, 1.5), IMM)
    for _ in range(5):
        dot(px + pw * .70 + rng.normal(0, 1.5), py + ph * .58 + rng.normal(0, 1.3),
            TUM, sq=True, s=.8)


def self_int(px, py, pw, ph):
    yc = py + ph * .64
    a, b = px + pw * .26, px + pw * .74
    dot(a, yc, IMM); dot(b, yc, IMM)
    ax.annotate("", xy=(b - 1.4, yc), xytext=(a + 1.4, yc),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=.9,
                                mutation_scale=7))


def cross_int(px, py, pw, ph):
    yc = py + ph * .64
    a, b = px + pw * .26, px + pw * .74
    dot(a, yc, IMM); dot(b, yc, TUM, sq=True, s=.85)
    ax.annotate("", xy=(b - 1.5, yc), xytext=(a + 1.4, yc),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=.9,
                                mutation_scale=7))


def infiltration(px, py, pw, ph):
    yc = py + ph * .64
    for _ in range(6):
        dot(px + pw * .22 + rng.normal(0, 1.2), yc + rng.normal(0, 1.5), IMM, s=.7)
    for _ in range(6):
        dot(px + pw * .76 + rng.normal(0, 1.2), yc + rng.normal(0, 1.5), TUM,
            sq=True, s=.7)
    ax.annotate("", xy=(px + pw * .62, yc), xytext=(px + pw * .38, yc),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2,
                                mutation_scale=9))


def modularity(px, py, pw, ph):
    A = np.array([[.22, .50], [.30, .82], [.38, .50]])
    B = np.array([[.64, .82], [.72, .48], [.80, .80]])
    for grp, col, sq in ((A, IMM, False), (B, TUM, True)):
        P = np.column_stack([px + grp[:, 0] * pw, py + grp[:, 1] * ph])
        for m in range(3):
            for n in range(m + 1, 3):
                ax.plot(P[[m, n], 0], P[[m, n], 1], lw=.7, c="#9AA3AA", zorder=3)
        for p in P:
            dot(p[0], p[1], col, sq=sq, s=.8)
    ax.plot([px + .38 * pw, px + .64 * pw], [py + .50 * ph, py + .82 * ph],
            lw=.7, c="#9AA3AA", ls=(0, (2, 2)), zorder=2)


def ripley(px, py, pw, ph):
    cxx, cyy = px + pw * .5, py + ph * .64
    for r in (1.6, 2.6, 3.6, 4.6):
        ax.add_patch(Circle((cxx, cyy), r, fc="none", ec="#9AA3AA", lw=.6,
                            ls=(0, (2, 2)), zorder=2))
    dot(cxx, cyy, IMM, s=.85)
    for t in np.linspace(0, 2 * np.pi, 8)[:-1]:
        rr = 1.3 + rng.random() * 3.2
        dot(cxx + rr * np.cos(t), cyy + rr * np.sin(t), IMM, s=.65)


FEATS = [(abundance,    r"$N_A,\ f_A$",             "abundance"),
         (self_int,     r"$I_{A\rightarrow A}$",     "self-interaction"),
         (cross_int,    r"$I_{A\rightarrow B}$",     "interaction"),
         (infiltration, r"$F_{A\rightarrow B}$",     "infiltration"),
         (modularity,   r"$M$",                      "modularity"),
         (ripley,       r"$C_A(r)$",                 "clustering")]

for k, (fn, sym, name) in enumerate(FEATS):
    px, py = COLX[k % 2], ROWY[k // 2]
    ax.add_patch(Rectangle((px, py), cw, ch, fc="#FCFCFD", ec=PALE, lw=.9,
                           zorder=2))
    fn(px, py, cw, ch)
    ax.text(px + cw / 2, py + 2.7, name, ha="center", va="bottom",
            fontsize=5.9, color=INK, zorder=5)
    ax.text(px + cw / 2, py + 0.7, sym, ha="center", va="bottom",
            fontsize=6.4, color=GREY, zorder=5)

# ------------------------------------------------- 5. spatial state tiles
x, w = STAGE[4][1], STAGE[4][2]
tw = w - 4.0
BLOCK = (TOP - BOT - 2.0) / 3.0
TH = BLOCK - 3.2
LABELS = ["distant co-occurrence", "bordering", "infiltrated"]


def tile(px, py, pw, ph, kind):
    ax.add_patch(Rectangle((px, py), pw, ph, fc="#FCFCFD", ec=PALE, lw=.9,
                           zorder=2))
    lo = np.array([px + 1.0, py + 1.0])
    hi = np.array([px + pw - 1.0, py + ph - 1.0])

    if kind == "distant":
        I = np.column_stack([px + pw * .26 + rng.normal(0, 2.4, 15),
                             py + ph * .60 + rng.normal(0, 1.4, 15)])
        T = np.column_stack([px + pw * .72 + rng.normal(0, 2.4, 13),
                             py + ph * .38 + rng.normal(0, 1.4, 13)])
    elif kind == "bordering":
        t = rng.uniform(0.12, np.pi - 0.12, 24)
        r = 1.0 + rng.random(24) ** 1.4 * 2.2
        I = np.column_stack([px + pw * .5 + r * np.cos(t) * 3.6,
                             py + ph * .26 + r * np.sin(t) * 2.0])
        T = np.column_stack([px + pw * .5 + rng.normal(0, 1.0, 8),
                             py + ph * .24 + rng.normal(0, .55, 8)])
    else:
        I = np.column_stack([px + rng.uniform(.08, .92, 16) * pw,
                             py + rng.uniform(.16, .84, 16) * ph])
        T = np.column_stack([px + rng.uniform(.08, .92, 13) * pw,
                             py + rng.uniform(.16, .84, 13) * ph])
    for p in np.clip(I, lo, hi):
        dot(p[0], p[1], IMM, s=.55)
    for p in np.clip(T, lo, hi):
        dot(p[0], p[1], TUM, sq=True, s=.55)


for k, kind in enumerate(["distant", "bordering", "infiltrated"]):
    ytop = TOP - 1.0 - k * BLOCK
    py = ytop - TH
    tile(x + 2.0, py, tw, TH, kind)
    ax.text(x + 2.0 + tw / 2, py - 0.7, LABELS[k], ha="center", va="top",
            fontsize=6.6, color=GREY, zorder=4)

for ext in ("pdf", "png"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / f"fig_flow2.{ext}", dpi=400, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
print("wrote fig_flow2.pdf and fig_flow2.png")
