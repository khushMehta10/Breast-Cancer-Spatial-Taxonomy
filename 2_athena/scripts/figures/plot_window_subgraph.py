"""
Standalone: window subgraph figure with true-size spots.

Change SLIDE_ID (and the window) at the top; everything else auto-detects:
  - which athena files to read
  - the platform (Spatial Transcriptomics / Visium / Xenium)
  - the spot diameter for that platform
  - the pixel size (mpp) for that slide
  - the spatial library key in the h5ad
"""

# ============================================================
# SETUP
# ============================================================

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle
from scipy import sparse
from scipy.sparse import triu
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

# ---- settings ----
SLIDE_ID = "SPA51"
SELECTED_WINDOW = "SPA51_r5_c10"

ST_DIR = config.BREAST_ST_DIR
ATHENA_DIR = config.ATHENA_DIR
OUTPUT_ATHENA_DIR = config.ATHENA_OUT


# ---- platform -> spot diameter (um) ----
def spot_diameter_um(technology):
    t = str(technology).lower()
    if "xenium" in t:
        return None  # subcellular, no capture spots
    if "visium" in t:
        return 55.0  # Visium: 55 um spots
    if "transcriptomics" in t or t.strip() == "st":
        return 100.0  # legacy Spatial Transcriptomics: 100 um
    return None  # unknown platform -> skip spots safely


# ---- load tables ----
cell_df = pd.read_csv(ATHENA_DIR / "cell_df" / f"{SLIDE_ID}_cells.csv")
spot_df = pd.read_csv(ATHENA_DIR / "spot_df" / f"{SLIDE_ID}_spots.csv")
window_df = pd.read_csv(ATHENA_DIR / "window_df" / f"{SLIDE_ID}_windows.csv")

# ---- platform + pixel size, straight from this slide's metadata ----
technology = window_df["technology"].iloc[0]
mpp = float(window_df["pixel_size_um_x"].iloc[0])  # microns per pixel
spot_um = spot_diameter_um(technology)
spot_radius_px = (spot_um / 2.0 / mpp) if spot_um else None

print(f"slide:      {SLIDE_ID}")
print(f"technology: {technology}")
print(f"mpp:        {mpp:.4f} um/px")
print(
    f"spot:       {spot_um} um  (radius {spot_radius_px:.1f} px)"
    if spot_um
    else "spot:       none for this platform"
)

# ---- tissue image + display scale (auto-detect the library key) ----
adata = ad.read_h5ad(ST_DIR / f"{SLIDE_ID}.h5ad")
spatial_library_key = list(adata.uns["spatial"].keys())[0]
spatial_library = adata.uns["spatial"][spatial_library_key]
tissue_image = np.asarray(spatial_library["images"]["downscaled_fullres"])
display_scale = float(
    spatial_library["scalefactors"]["tissue_downscaled_fullres_scalef"]
)

# ---- select the window and pull its subgraph ----
selected_window_id = SELECTED_WINDOW
selected_window = window_df.loc[window_df["window_id"] == selected_window_id].iloc[0]

selected_node_indices = cell_df.loc[
    cell_df["window_id"] == selected_window_id, "node_index"
].to_numpy(dtype=np.int64)

full_adjacency = sparse.load_npz(ATHENA_DIR / "cell_graphs" / f"{SLIDE_ID}_knn_k5.npz")
window_subgraph = full_adjacency[selected_node_indices][:, selected_node_indices]
n_edges = int(window_subgraph.nnz // 2)

print(f"window:     {selected_window_id}")
print(f"cells:      {len(selected_node_indices):,}")
print(f"edges:      {n_edges:,}")


# ============================================================
# SUBGRAPH FIGURE
# ============================================================

print("=" * 70)
print("PREPARING WINDOW GRAPH VISUALIZATION")
print("=" * 70)

selected_cells = cell_df.loc[cell_df["window_id"] == selected_window_id].copy()

selected_spots = spot_df.loc[spot_df["window_id"] == selected_window_id].copy()

print(f"Selected window ID:       {selected_window_id}")
print(f"Selected cells:           {len(selected_cells):,}")
print(f"Selected ST observations: {len(selected_spots):,}")
print(f"Subgraph shape:           {window_subgraph.shape}")


# validation
expected_cell_count = int(selected_window["n_cells"])
expected_spot_count = int(selected_window["n_molecular_observations"])

print("\n" + "=" * 70)
print("SELECTED DATA VALIDATION")
print("=" * 70)
print(f"Expected cells from window_df: {expected_cell_count:,}")
print(f"Expected spots from window_df: {expected_spot_count:,}")

if len(selected_cells) != expected_cell_count:
    raise ValueError("Selected cell count does not match window_df.")
if len(selected_spots) != expected_spot_count:
    raise ValueError("Selected ST observation count does not match window_df.")
if window_subgraph.shape != (len(selected_cells), len(selected_cells)):
    raise ValueError("Subgraph dimensions do not match selected cells.")
print("Selected cells, spots, and subgraph are consistent.")


# node-order check
current_selected_node_indices = selected_cells["node_index"].to_numpy(dtype=np.int64)
node_order_matches = np.array_equal(
    current_selected_node_indices, selected_node_indices
)
print("\n" + "=" * 70)
print("SUBGRAPH NODE ORDER VALIDATION")
print("=" * 70)
print(f"Selected cell order matches subgraph order: {node_order_matches}")
if not node_order_matches:
    raise ValueError(
        "selected_cells order does not match the window_subgraph node order."
    )


# extract unique undirected edges
print("\n" + "=" * 70)
print("EXTRACTING EDGES FOR VISUALIZATION")
print("=" * 70)
upper_triangle = triu(window_subgraph, k=1).tocsr()
edge_row_indices, edge_col_indices = upper_triangle.nonzero()
n_visualized_edges = len(edge_row_indices)
print(f"Unique undirected edges extracted: {n_visualized_edges:,}")


# full-resolution coordinates
selected_xy = selected_cells[["x", "y"]].to_numpy(dtype=float)
edge_segments = np.stack(
    [selected_xy[edge_row_indices], selected_xy[edge_col_indices]], axis=1
)

# to display coordinates
edge_segments_display = edge_segments * display_scale
selected_x_display = selected_cells["x"].to_numpy(dtype=float) * display_scale
selected_y_display = selected_cells["y"].to_numpy(dtype=float) * display_scale
selected_spot_x_display = selected_spots["x"].to_numpy(dtype=float) * display_scale
selected_spot_y_display = selected_spots["y"].to_numpy(dtype=float) * display_scale


# ---- adaptive marker sizing (readability only; keyed to cell count) ----
n_cells_win = len(selected_cells)
scale = (700.0 / max(n_cells_win, 1)) ** 0.5
scale = min(max(scale, 0.25), 3.0)
cell_base = 45.0 * scale
edge_width = max(0.4, 1.4 * scale)
edge_alpha = min(0.75, 0.35 + 0.4 * scale)

cell_type_styles = {
    "Neoplastic": {"color": "#FF0000", "size": cell_base * 1.00},  # (255,0,0)
    "Inflammatory": {"color": "#22DD4D", "size": cell_base * 1.00},  # (34,221,77)
    "Connective": {"color": "#235CEC", "size": cell_base * 0.70},  # (35,92,236)
    "Epithelial": {"color": "#FF9F44", "size": cell_base * 0.85},  # (255,159,68)
}
known_cell_types = set(cell_type_styles.keys())
observed_cell_types = set(selected_cells["cell_type"].dropna().unique())
unexpected_cell_types = observed_cell_types - known_cell_types


# ---- figure ----
print("\n" + "=" * 70)
print("CREATING SUBGRAPH FIGURE")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(tissue_image)

# edges
edge_collection = LineCollection(
    edge_segments_display, linewidths=edge_width, color="0.25", alpha=edge_alpha
)
ax.add_collection(edge_collection)

# cells by type
for cell_type, style in cell_type_styles.items():
    mask = selected_cells["cell_type"].to_numpy() == cell_type
    n_type_cells = int(mask.sum())
    if n_type_cells == 0:
        continue
    ax.scatter(
        selected_x_display[mask],
        selected_y_display[mask],
        s=style["size"],
        c=style["color"],
        alpha=0.80,
        label=f"{cell_type} (n={n_type_cells:,})",
        zorder=3,
    )

for cell_type in sorted(unexpected_cell_types):
    mask = selected_cells["cell_type"].to_numpy() == cell_type
    n_type_cells = int(mask.sum())
    ax.scatter(
        selected_x_display[mask],
        selected_y_display[mask],
        s=cell_base * 0.8,
        alpha=0.80,
        label=f"{cell_type} (n={n_type_cells:,})",
        zorder=3,
    )

# ---- ST spots at true physical size (Circle patches in data coords) ----
if spot_radius_px is not None:
    r_disp = spot_radius_px * display_scale
    for sx, sy in zip(selected_spot_x_display, selected_spot_y_display):
        ax.add_patch(
            Circle(
                (sx, sy),
                radius=r_disp,
                facecolor="none",
                edgecolor="black",
                linewidth=1.3,
                zorder=4,
            )
        )
    # legend proxy
    ax.scatter(
        [],
        [],
        s=60,
        facecolors="none",
        edgecolors="black",
        linewidths=1.3,
        label=f"ST observations (n={len(selected_spots):,})",
    )
    print(
        f"Drew {len(selected_spots):,} spots at {spot_um:.0f} um "
        f"(radius {r_disp:.1f} display px)"
    )
else:
    print("No capture spots for this platform; spots not drawn.")

# window boundary
window_rectangle = Rectangle(
    (selected_window["xmin"] * display_scale, selected_window["ymin"] * display_scale),
    width=(selected_window["xmax"] - selected_window["xmin"]) * display_scale,
    height=(selected_window["ymax"] - selected_window["ymin"]) * display_scale,
    fill=False,
    linewidth=2.0,
    edgecolor="black",
    zorder=5,
)
ax.add_patch(window_rectangle)

# zoom to the window with a small margin
margin_fullres_px = 100
x_min_display = (selected_window["xmin"] - margin_fullres_px) * display_scale
x_max_display = (selected_window["xmax"] + margin_fullres_px) * display_scale
y_min_display = (selected_window["ymin"] - margin_fullres_px) * display_scale
y_max_display = (selected_window["ymax"] + margin_fullres_px) * display_scale
ax.set_xlim(x_min_display, x_max_display)
ax.set_ylim(y_max_display, y_min_display)

ax.set_title(
    f"{SLIDE_ID} \u2014 {selected_window_id}\n"
    f"{len(selected_cells):,} cells, {n_visualized_edges:,} internal edges, "
    f"{len(selected_spots):,} ST observations"
)
ax.set_xlabel("Downscaled image x")
ax.set_ylabel("Downscaled image y")
ax.legend(loc="upper right", fontsize=8, frameon=True)

# save
FIGURE_DIR = OUTPUT_ATHENA_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
subgraph_figure_path = FIGURE_DIR / f"{selected_window_id}_subgraph.png"
plt.tight_layout()
plt.savefig(subgraph_figure_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {subgraph_figure_path}")
