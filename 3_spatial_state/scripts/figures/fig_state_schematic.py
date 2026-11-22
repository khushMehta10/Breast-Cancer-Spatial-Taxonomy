"""
Illustration of the three interaction states.

Each panel is a 500 um window. Immune and tumor nuclei are drawn at the MEDIAN
counts observed for that state across the cohort:
  distant co-occurrence  207 immune, 236 tumor
  bordering              328 immune,  10 tumor
  infiltrated             32 immune,  11 tumor
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

IMM, TUM = "#0072B2", "#D55E00"          # Okabe-Ito blue / vermilion
rng = np.random.default_rng(3)
W = 500.0

def blob(n, cx, cy, sd):
    return np.clip(rng.normal([cx, cy], sd, size=(n, 2)), 10, W - 10)

def spread(n):
    return rng.uniform(22, 478, size=(n, 2))

PANELS = [
    ("Distant co-occurrence", "3.35",
     np.vstack([blob(140, 145, 340, 60), blob(67, 110, 300, 38)]),
     np.vstack([blob(160, 358, 158, 60), blob(76, 392, 192, 38)]),
     ["interaction below chance",
      "both compartments clustered",
      "modularity high"]),
    ("Bordering", "2.78",
     blob(328, 250, 322, 94),
     blob(10, 250, 88, 32),
     ["interaction above chance",
      "immune$\\rightarrow$tumor infiltration low",
      "immune clustered, $\\geq$5 tumor nuclei",
      "modularity high"]),
    ("Infiltrated", "5.49",
     spread(32), spread(11),
     ["interaction above chance",
      "infiltration high both directions",
      "modularity low"]),
]

fig, axes = plt.subplots(1, 3, figsize=(10.4, 5.0))
for ax, (name, ratio, imm, tum, cond) in zip(axes, PANELS):
    ax.add_patch(Rectangle((0, 0), W, W, fc="#FCFCFD", ec="#2B2B2B", lw=1.2,
                           zorder=0))
    ax.scatter(imm[:, 0], imm[:, 1], s=13, c=IMM, alpha=.88, lw=0, zorder=3)
    ax.scatter(tum[:, 0], tum[:, 1], s=34, c=TUM, alpha=.95, lw=0, zorder=4,
               marker="s")

    # 100 um scale bar
    ax.plot([28, 128], [26, 26], lw=3.2, c="#2B2B2B", zorder=6,
            solid_capstyle="butt")
    ax.text(78, 44, "100 $\\mu$m", ha="center", va="bottom", fontsize=7.2,
            color="#2B2B2B", zorder=6)

    ax.set_xlim(-14, W + 14); ax.set_ylim(-14, W + 14)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{name}\n{ratio}$\\times$ over chance", fontsize=11.5,
                 pad=9, linespacing=1.5)
    LH = 0.055                      # one text line, as a fraction of axes height
    ax.text(0.5, -0.05, "\n".join(cond), transform=ax.transAxes, ha="center",
            va="top", fontsize=7.8, color="#444444", linespacing=1.55)
    ax.text(0.5, -0.05 - LH * len(cond) - 0.03,
            f"median {len(imm)} immune, {len(tum)} tumor",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.4,
            color="#111111")

h = [plt.Line2D([], [], marker="o", ls="", color=IMM, ms=7, label="Immune nucleus"),
     plt.Line2D([], [], marker="s", ls="", color=TUM, ms=7, label="Tumor nucleus")]
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.40, wspace=0.09)
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.015))
config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
plt.savefig(config.FIGURES_OUT / "fig_states.pdf", bbox_inches="tight")
plt.savefig(config.FIGURES_OUT / "fig_states.png", dpi=300, bbox_inches="tight")
print("wrote fig_states.pdf and fig_states.png")
