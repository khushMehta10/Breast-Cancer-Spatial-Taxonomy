"""Results figure: one window per interaction state, at nucleus resolution.

Layout
  (a)        whole slide, every labelled window tinted by state, the three
             windows shown at right outlined and lettered
  (b)(c)(d)  one 500 um window per state, read from the pyramidal WSI, with
             a CellViT contour on every immune and tumor nucleus

The three windows are the same physical size, and each frame color matches
its window in (a).

Immune bluish green, tumor vermilion. Connective, epithelial and dead are
not drawn, since the figure is about the immune-tumor pair; set SHOW_OTHER
to include them.

  conda activate hest
  python 3_spatial_state/scripts/figures/fig_state_overlay.py
"""

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,      # embed as TrueType, not Type 3
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

# ============================================================ config
SLIDE = "SPA106"
CELL_FILE = None        # set a path to skip discovery
WSI_FILE = None         # set a path to skip discovery
WINDOW_UM = 500.0
PAD_FRAC = 0.04         # margin around each window in its zoom panel
TARGET_PX = 1400        # pixels across for each zoom panel
SHOW_OTHER = False
PICK = "cells"          # "cells" = window with most immune+tumor nuclei

ST_DIR = config.BREAST_ST_DIR
STEP5 = config.STEP5_TABLES
SEARCH_DIRS = [config.DATA_ROOT, config.REPO_ROOT]
OUT_DIR = config.FIGURES_OUT

IMM, TUM, OTH = "#009E73", "#D55E00", "#B9C0C7"
ORDER = ["distant_co_occurrence", "bordering", "infiltrated"]
STATE_COLOR = {"distant_co_occurrence": "#4C6E8A",
               "bordering": "#E4A11B",
               "infiltrated": "#35B779"}
STATE_LABEL = {"distant_co_occurrence": "distant co-occurrence",
               "bordering": "bordering", "infiltrated": "infiltrated"}
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

ID_TO_CLASS = {0: "Neoplastic", 1: "Inflammatory", 2: "Connective",
               3: "Dead", 4: "Epithelial"}
ALIAS = {"neoplastic": "Neoplastic", "tumor": "Neoplastic",
         "tumour": "Neoplastic", "inflammatory": "Inflammatory",
         "immune": "Inflammatory", "lymphocyte": "Inflammatory",
         "connective": "Connective", "stroma": "Connective",
         "stromal": "Connective", "epithelial": "Epithelial",
         "dead": "Dead", "necrosis": "Dead", "necrotic": "Dead"}
CONTOUR_KEYS = ["contour", "contours", "polygon", "coords", "coordinates",
                "geometry", "outline", "boundary"]
TYPE_KEYS = ["type", "cell_type", "class", "classification", "label",
             "type_name", "pannuke_type", "phenotype"]
X_KEYS = ["x", "centroid_x", "cx", "x_centroid", "global_x"]
Y_KEYS = ["y", "centroid_y", "cy", "y_centroid", "global_y"]


# ============================================================ readers
def norm_class(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "Unclassified"
    if isinstance(v, (int, np.integer)) or (
            isinstance(v, float) and float(v).is_integer()):
        return ID_TO_CLASS.get(int(v), f"type_{int(v)}")
    return ALIAS.get(str(v).strip().lower(), str(v).strip())


def as_ring(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    if isinstance(obj, dict):
        if obj.get("type") == "Polygon":
            obj = obj["coordinates"][0]
        elif obj.get("type") == "MultiPolygon":
            obj = obj["coordinates"][0][0]
        else:
            return None
    a = np.asarray(obj, dtype=float)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] != 2:
        return None
    return a


def first_key(d, keys):
    lower = {str(k).lower(): k for k in d}
    for k in keys:
        if k in d:
            return k
        if k in lower:
            return lower[k]
    return None


def discover(patterns, what):
    hits = []
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for p in patterns:
            hits += [Path(f) for f in glob.glob(str(d / p), recursive=True)]
    hits = [h for h in sorted(set(hits), key=lambda f: -f.stat().st_size)
            if h.is_file()]
    if not hits:
        return None
    print(f"{what} candidates:")
    for h in hits[:6]:
        print(f"  {h.stat().st_size/1e6:9.1f} MB  {h}")
    print(f"using {hits[0]}")
    return hits[0]


def load_cells(path):
    if path.suffix in (".json", ".geojson"):
        with open(path) as fh:
            obj = json.load(fh)
        out = []
        if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
            for ft in obj["features"]:
                p = ft.get("properties", {}) or {}
                k = first_key(p, TYPE_KEYS)
                cls = p[k] if k else None
                if isinstance(cls, dict):
                    cls = cls.get("name")
                out.append((as_ring(ft.get("geometry")), norm_class(cls)))
            return out
        if isinstance(obj, dict) and "nuc" in obj:
            for _, rec in obj["nuc"].items():
                ck = first_key(rec, CONTOUR_KEYS)
                tk = first_key(rec, TYPE_KEYS)
                out.append((as_ring(rec.get(ck)) if ck else None,
                            norm_class(rec.get(tk) if tk else None)))
            return out
        if isinstance(obj, list):
            for rec in obj:
                ck = first_key(rec, CONTOUR_KEYS)
                tk = first_key(rec, TYPE_KEYS)
                out.append((as_ring(rec.get(ck)) if ck else None,
                            norm_class(rec.get(tk) if tk else None)))
            return out
        sys.exit(f"unrecognised JSON in {path}")
    df = (pd.read_parquet(path) if path.suffix == ".parquet"
          else pd.read_csv(path))
    cols = {c.lower(): c for c in df.columns}
    ck = next((cols[k] for k in CONTOUR_KEYS if k in cols), None)
    tk = next((cols[k] for k in TYPE_KEYS if k in cols), None)
    xk = next((cols[k] for k in X_KEYS if k in cols), None)
    yk = next((cols[k] for k in Y_KEYS if k in cols), None)
    if tk is None:
        sys.exit("no cell-type column found")
    out = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        ring = as_ring(d[ck]) if ck else None
        if ring is None and xk and yk:
            a = np.linspace(0, 2 * np.pi, 14, endpoint=False)
            ring = np.column_stack([float(d[xk]) + 6 * np.cos(a),
                                    float(d[yk]) + 6 * np.sin(a)])
        out.append((ring, norm_class(d[tk])))
    return out


def read_region(path, box, target_px):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    want = max(1.0, (x1 - x0) / float(target_px))
    try:
        import openslide
        sl = openslide.OpenSlide(str(path))
        lvl = sl.get_best_level_for_downsample(want)
        ds = sl.level_downsamples[lvl]
        im = sl.read_region((x0, y0), lvl,
                            (int((x1 - x0) / ds), int((y1 - y0) / ds)))
        return np.asarray(im.convert("RGB")), 1.0 / ds
    except Exception as e:
        print(f"  openslide unavailable ({e}); using tifffile")
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        levels = series.levels if hasattr(series, "levels") else [series]
        best, bds, w0 = 0, 1.0, levels[0].shape[1]
        for i, lv in enumerate(levels):
            ds = w0 / lv.shape[1]
            if ds <= want:
                best, bds = i, ds
        arr = levels[best].asarray()
    sub = arr[int(y0 / bds):int(y1 / bds), int(x0 / bds):int(x1 / bds)]
    if sub.ndim == 3 and sub.shape[2] == 4:
        sub = sub[..., :3]
    return sub, 1.0 / bds


# ============================================================ inputs
tbl = pd.read_csv(STEP5 / f"{SLIDE}_interaction_state.csv")
st = tbl["interaction_state"].astype(str)
lab = tbl[st.isin(STATE_COLOR)].copy()
print(f"{SLIDE}: {len(tbl):,} windows, {len(lab)} labelled")
print(lab["interaction_state"].value_counts().to_string())
if lab.empty:
    sys.exit("no labelled windows; pick another slide")

h5 = ST_DIR / f"{SLIDE}.h5ad"
if not h5.exists():
    sys.exit(f"no {SLIDE}.h5ad found")
adata = ad.read_h5ad(h5, backed="r")
libr = adata.uns["spatial"][list(adata.uns["spatial"].keys())[0]]
img = np.asarray(libr["images"]["downscaled_fullres"])
SCALE = float(libr["scalefactors"]["tissue_downscaled_fullres_scalef"])
if img.dtype != np.uint8:
    img = (255 * img.astype(float) / max(1.0, img.max())).astype(np.uint8)
if img.ndim == 3 and img.shape[2] == 4:
    img = img[..., :3]
IH, IW = img.shape[:2]
win_px = float(np.median(tbl.xmax - tbl.xmin))
UM_PER_PX = WINDOW_UM / win_px

wsi = Path(WSI_FILE) if WSI_FILE else discover(
    [f"**/wsis/{SLIDE}.tif*", f"**/{SLIDE}.tif", f"**/{SLIDE}.svs",
     f"**/{SLIDE}.ndpi"], "WSI")
if wsi is None:
    sys.exit("no pyramidal WSI found; the thumbnail is too coarse for nuclei")

src = Path(CELL_FILE) if CELL_FILE else discover(
    [f"**/{SLIDE}*cell*.geojson", f"**/{SLIDE}*cell*.json",
     f"**/{SLIDE}*_cells.*", f"**/{SLIDE}*cell*.csv"], "cell file")
if src is None:
    sys.exit("no cell file found")
cells = load_cells(src)
print(f"{len(cells):,} nuclei read")

# ============================================================ pick windows
chosen = []
for state in ORDER:
    sub = lab[lab.interaction_state == state]
    if sub.empty:
        print(f"  no {state} window on this slide; panel skipped")
        continue
    if PICK == "cells" and {"inflammatory_count", "neoplastic_count"} <= set(sub.columns):
        k = (sub.inflammatory_count.fillna(0) + sub.neoplastic_count.fillna(0)).idxmax()
    else:
        k = sub.index[0]
    r = sub.loc[k]
    chosen.append((state, r))
    print(f"  {state}: window {r.get('window_id', k)}, "
          f"{int(r.get('inflammatory_count', -1))} immune, "
          f"{int(r.get('neoplastic_count', -1))} tumor")
if not chosen:
    sys.exit("nothing to draw")

# ============================================================ figure
n = len(chosen)
FIGW = 6.5
PAD_L, GAP_A, GAP_Z, PAD_R = 0.03, 0.16, 0.06, 0.03   # inches
TITLE_H, LEG_H, TOP_PAD = 0.17, 0.26, 0.02

ASP = IW / float(IH)                       # panel (a) width per unit height
T = (FIGW - PAD_L - GAP_A - (n - 1) * GAP_Z - PAD_R) / (n + ASP)
FIGH = TOP_PAD + TITLE_H + T + LEG_H
fig = plt.figure(figsize=(FIGW, FIGH))


def rect(x_in, w_in):
    """axes rect for a panel of height T sitting on the legend strip"""
    return [x_in / FIGW, LEG_H / FIGH, w_in / FIGW, T / FIGH]


axa = fig.add_axes(rect(PAD_L, T * ASP))
axa.imshow(img)
for _, r in lab.iterrows():
    c = STATE_COLOR[str(r.interaction_state)]
    axa.add_patch(Rectangle((r.xmin * SCALE, r.ymin * SCALE),
                            (r.xmax - r.xmin) * SCALE,
                            (r.ymax - r.ymin) * SCALE,
                            fc=c, ec="white", alpha=.75, lw=.6, zorder=3))
axa.set_xlim(0, IW); axa.set_ylim(IH, 0)
axa.set_xticks([]); axa.set_yticks([])
for sp in axa.spines.values():
    sp.set_linewidth(0.7); sp.set_color("#6E7479")
axa.text(0.03, 0.97, "a", transform=axa.transAxes, ha="left", va="top",
         fontsize=9, fontweight="bold", color="#111111", path_effects=HALO)

letters = "bcd"
for i, (state, r) in enumerate(chosen):
    w = r.xmax - r.xmin
    pad = PAD_FRAC * w
    box = (r.xmin - pad, r.ymin - pad, r.xmax + pad, r.ymax + pad)
    tile, SB = read_region(wsi, box, TARGET_PX)
    OXX, OYY = box[0] * SB, box[1] * SB

    polys, colors = [], []
    for ring, cls in cells:
        if ring is None:
            continue
        cx, cy = ring[:, 0].mean(), ring[:, 1].mean()
        if not (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
            continue
        if cls == "Inflammatory":
            col = IMM
        elif cls == "Neoplastic":
            col = TUM
        elif SHOW_OTHER:
            col = OTH
        else:
            continue
        polys.append(np.column_stack([ring[:, 0] * SB - OXX,
                                      ring[:, 1] * SB - OYY]))
        colors.append(col)

    x0 = PAD_L + T * ASP + GAP_A + i * (T + GAP_Z)
    axz = fig.add_axes(rect(x0, T))
    axz.imshow(tile)
    if polys:
        axz.add_collection(PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolors=[matplotlib.colors.to_rgba(c, .32) for c in colors],
            edgecolors=colors, linewidths=.5, zorder=3))
    axz.set_xlim(0, tile.shape[1]); axz.set_ylim(tile.shape[0], 0)
    axz.set_xticks([]); axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_linewidth(1.4); sp.set_color(STATE_COLOR[state])
    axz.text(0.5, 1.02, STATE_LABEL[state], transform=axz.transAxes,
             ha="center", va="bottom", fontsize=7.4, color="#111111")
    axz.text(0.035, 0.965, letters[i], transform=axz.transAxes, ha="left",
             va="top", fontsize=9, fontweight="bold", color="#111111",
             path_effects=HALO)
    if i == 0:
        bar_um = 100.0
        bar_px = bar_um / UM_PER_PX * SB
        bx, by = .05 * tile.shape[1], .945 * tile.shape[0]
        axz.plot([bx, bx + bar_px], [by, by], lw=2.6, c="#111111", zorder=6,
                 solid_capstyle="butt", path_effects=HALO)
        axz.text(bx + bar_px / 2, by - .015 * tile.shape[0],
                 f"{bar_um:.0f} $\\mu$m", ha="center", va="bottom",
                 fontsize=6.6, color="#111111", zorder=6, path_effects=HALO)

    # mark this window in panel (a)
    axa.add_patch(Rectangle((r.xmin * SCALE, r.ymin * SCALE), w * SCALE,
                            (r.ymax - r.ymin) * SCALE, fc="none",
                            ec="#111111", lw=1.2, zorder=5,
                            path_effects=HALO))
    axa.text((r.xmin + w * 0.5) * SCALE, (r.ymin - w * 0.10) * SCALE,
             letters[i], ha="center", va="bottom", fontsize=7.5,
             fontweight="bold", color="#111111", zorder=6,
             path_effects=HALO)

h = [plt.Line2D([], [], marker="o", ls="",
                mfc=matplotlib.colors.to_rgba(c, .35), mec=c, mew=1.1, ms=6,
                label=t) for c, t in
     ((IMM, "immune nucleus"), (TUM, "tumor nucleus"))]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False,
           fontsize=7.2, bbox_to_anchor=(.5, .002), handletextpad=.35,
           columnspacing=1.8)

for ext in ("pdf", "png"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / f"fig_states_{SLIDE}.{ext}", dpi=400, facecolor="white",
                bbox_inches="tight", pad_inches=.03)
print(f"wrote fig_states_{SLIDE}.pdf and .png")
