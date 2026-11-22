"""
Standalone: local Shannon entropy figure (spatial map + distribution).
Reads cell_df + cell_graph, computes local Shannon entropy per cell, and plots it.
"""

# ============================================================
# SETUP
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

SLIDE_ID           = "SPA51"
selected_window_id = "SPA51_r5_c10"

ATHENA_DIR = config.ATHENA_DIR
OUTPUT_ATHENA_DIR = config.ATHENA_OUT

cell_df    = pd.read_csv(ATHENA_DIR / "cell_df" / f"{SLIDE_ID}_cells.csv")
cell_graph = sparse.load_npz(ATHENA_DIR / "cell_graphs" / f"{SLIDE_ID}_knn_k5.npz").tocsr()

print(f"slide:  {SLIDE_ID}")
print(f"window: {selected_window_id}")
print(f"cells:  {len(cell_df):,}   graph: {cell_graph.shape}")


# ============================================================
# COMPUTE  -- local Shannon entropy per cell
# ============================================================
cell_types_array = cell_df["cell_type"].to_numpy()
full_shannon_values = np.zeros(len(cell_df), dtype=np.float64)

for node in range(len(cell_df)):
    neighbor_indices = cell_graph[node].indices
    neighborhood_indices = np.append(neighbor_indices, node)
    neighborhood_types = cell_types_array[neighborhood_indices]

    unique_types, type_counts = np.unique(neighborhood_types, return_counts=True)
    type_proportions = type_counts / type_counts.sum()
    shannon_entropy = -np.sum(type_proportions * np.log(type_proportions))

    full_shannon_values[node] = shannon_entropy

selected_mask    = cell_df["window_id"].to_numpy() == selected_window_id
selected_indices = np.where(selected_mask)[0]
selected_cells   = cell_df.loc[selected_mask].copy()
local_shannon_values = full_shannon_values[selected_indices]

print(f"Cells in {selected_window_id}: {len(selected_indices):,}")


# ============================================================
# PLOT  -- two panels: spatial map + distribution
# ============================================================
fig, (ax_map, ax_hist) = plt.subplots(
    1,
    2,
    figsize=(16, 7),
)


# Left: spatial map colored by local Shannon entropy
scatter = ax_map.scatter(
    selected_cells["x"],
    selected_cells["y"],
    c=local_shannon_values,
    cmap="plasma",
    s=10,
    vmin=0.0,
    vmax=np.log(4),
)

ax_map.set_title(
    f"{selected_window_id}\n"
    "Local Shannon entropy"
)

ax_map.set_xlabel("x coordinate (pixels)")
ax_map.set_ylabel("y coordinate (pixels)")
ax_map.invert_yaxis()
ax_map.set_aspect("equal")

colorbar = fig.colorbar(
    scatter,
    ax=ax_map,
)

colorbar.set_label("Local Shannon entropy")


# Right: Shannon entropy distribution
ax_hist.hist(
    local_shannon_values,
    bins=40,
    color="steelblue",
    edgecolor="black",
    linewidth=0.5,
)

mean_shannon = local_shannon_values.mean()
median_shannon = np.median(local_shannon_values)

ax_hist.axvline(
    mean_shannon,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {mean_shannon:.3f}",
)

ax_hist.axvline(
    median_shannon,
    color="green",
    linestyle="--",
    linewidth=1.5,
    label=f"Median = {median_shannon:.3f}",
)

ax_hist.axvline(
    np.log(4),
    color="black",
    linestyle=":",
    linewidth=1.5,
    label=f"Max possible = {np.log(4):.3f}",
)

ax_hist.set_title(
    f"{selected_window_id}\n"
    "Local Shannon entropy distribution"
)

ax_hist.set_xlabel("Local Shannon entropy")
ax_hist.set_ylabel("Number of cells")
ax_hist.legend()


# Save
FIGURE_DIR = OUTPUT_ATHENA_DIR / "figures" / "local_shannon"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

figure_path = FIGURE_DIR / f"{selected_window_id}_local_shannon.png"

plt.tight_layout()
plt.savefig(figure_path, dpi=300, bbox_inches="tight")

print(f"Saved figure: {figure_path}")
