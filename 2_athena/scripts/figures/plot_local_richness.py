"""
Standalone: local phenotype richness figure (spatial map + distribution).
Reads cell_df + cell_graph, computes local richness per cell, and plots it.
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
# COMPUTE  -- local richness per cell
# ============================================================
cell_types_array = cell_df["cell_type"].to_numpy()
full_richness_values = np.zeros(len(cell_df), dtype=np.int64)

for node in range(len(cell_df)):
    neighbor_indices = cell_graph[node].indices
    neighborhood_indices = np.append(neighbor_indices, node)
    neighborhood_types = cell_types_array[neighborhood_indices]
    full_richness_values[node] = len(np.unique(neighborhood_types))

print("Local richness range (slide):",
      full_richness_values.min(), "to", full_richness_values.max())

selected_mask    = cell_df["window_id"].to_numpy() == selected_window_id
selected_indices = np.where(selected_mask)[0]
selected_cells   = cell_df.loc[selected_mask].copy()
local_richness_values = full_richness_values[selected_indices]

print(f"Cells in {selected_window_id}: {len(selected_indices):,}")

richness_values, richness_counts = np.unique(local_richness_values, return_counts=True)
print("Richness distribution:", dict(zip(richness_values.tolist(), richness_counts.tolist())))


# ============================================================
# PLOT  -- two panels: spatial map + distribution
# ============================================================
fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(16, 7))

scatter = ax_map.scatter(
    selected_cells["x"], selected_cells["y"],
    c=local_richness_values, cmap="viridis", s=10,
    vmin=1, vmax=max(4, local_richness_values.max()),
)
ax_map.set_title(f"{selected_window_id}\nLocal phenotype richness")
ax_map.set_xlabel("x coordinate (pixels)")
ax_map.set_ylabel("y coordinate (pixels)")
ax_map.invert_yaxis()
ax_map.set_aspect("equal")
cbar = fig.colorbar(scatter, ax=ax_map)
cbar.set_label("Local richness")

total_cells = len(local_richness_values)
richness_percentages = richness_counts / total_cells * 100.0
bar_colors = plt.cm.viridis((richness_values - 1) / max(3, richness_values.max() - 1))

bars = ax_hist.bar(richness_values, richness_percentages,
                   color=bar_colors, edgecolor="black", linewidth=0.5)
for bar, pct in zip(bars, richness_percentages):
    ax_hist.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{pct:.2f}%", ha="center", va="bottom")

ax_hist.set_title(f"{selected_window_id}\nLocal richness distribution")
ax_hist.set_xlabel("Local richness")
ax_hist.set_ylabel("Cells (%)")
ax_hist.set_xticks(richness_values)

FIGURE_DIR = OUTPUT_ATHENA_DIR / "figures" / "local_richness"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
figure_path = FIGURE_DIR / f"{selected_window_id}_local_richness.png"
plt.tight_layout()
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
print(f"Saved figure: {figure_path}")
