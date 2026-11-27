"""
Standalone: 500 um molecular window overlay figure.

Spots are drawn at their true physical diameter (Circle patches in data
coordinates), auto-detected per platform. Change SLIDE_ID; platform,
spot size, mpp and the spatial library key all resolve automatically.
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
from matplotlib.patches import Rectangle, Circle
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

SLIDE_ID = "SPA51"

ST_DIR = config.BREAST_ST_DIR
ATHENA_DIR = config.ATHENA_DIR
OUTPUT_ATHENA_DIR = config.ATHENA_OUT


# ---- platform -> spot diameter (um) ----
def spot_diameter_um(technology):
    t = str(technology).lower()
    if "xenium" in t:
        return None  # subcellular, no spots
    if "visium" in t:
        return 55.0
    if "transcriptomics" in t or t.strip() == "st":
        return 100.0
    return None


spot_df = pd.read_csv(ATHENA_DIR / "spot_df" / f"{SLIDE_ID}_spots.csv")
window_df = pd.read_csv(ATHENA_DIR / "window_df" / f"{SLIDE_ID}_windows.csv")

# platform + pixel size straight from this slide's metadata
technology = window_df["technology"].iloc[0]
mpp = float(window_df["pixel_size_um_x"].iloc[0])
spot_um = spot_diameter_um(technology)
spot_radius_px = (spot_um / 2.0 / mpp) if spot_um else None

adata = ad.read_h5ad(ST_DIR / f"{SLIDE_ID}.h5ad")
spatial_library_key = list(adata.uns["spatial"].keys())[0]
spatial_library = adata.uns["spatial"][spatial_library_key]

tissue_image = np.asarray(spatial_library["images"]["downscaled_fullres"])
display_scale = float(
    spatial_library["scalefactors"]["tissue_downscaled_fullres_scalef"]
)
image_height, image_width = tissue_image.shape[:2]

spot_x_display = spot_df["x"].to_numpy() * display_scale
spot_y_display = spot_df["y"].to_numpy() * display_scale

print(f"slide:        {SLIDE_ID}")
print(f"technology:   {technology}")
print(f"mpp:          {mpp:.4f} um/px")
print(
    f"spot:         {spot_um} um" if spot_um else "spot:         none for this platform"
)
print(f"library key:  {spatial_library_key}")
print(f"image shape:  {tissue_image.shape}")
print(f"scale:        {display_scale:.8f}")


# ============================================================
# WINDOW OVERLAY  -- spots at true physical size
# ============================================================

print("=" * 70)
print("VISUALIZING MOLECULAR WINDOWS")
print("=" * 70)
print(f"Number of molecular windows: {len(window_df):,}")

fig, ax = plt.subplots(figsize=(8, 12))
ax.imshow(tissue_image)

# window rectangles
for row in window_df.itertuples(index=False):
    rectangle = Rectangle(
        (row.xmin * display_scale, row.ymin * display_scale),
        width=(row.xmax - row.xmin) * display_scale,
        height=(row.ymax - row.ymin) * display_scale,
        fill=False,
        linewidth=0.7,
        edgecolor="blue",
    )
    ax.add_patch(rectangle)

# ST observations at their true physical diameter
if spot_radius_px is not None:
    r_disp = spot_radius_px * display_scale
    for x0, y0 in zip(spot_x_display, spot_y_display):
        ax.add_patch(
            Circle(
                (x0, y0),
                radius=r_disp,
                facecolor="none",
                edgecolor="black",
                linewidth=1.0,
            )
        )
    print(
        f"Drew {len(spot_df):,} spots at {spot_um:.0f} um "
        f"(radius {r_disp:.2f} display px)"
    )
else:
    # no capture spots (e.g. Xenium) -> small markers as a fallback
    ax.scatter(spot_x_display, spot_y_display, s=4, c="black", alpha=0.5)
    print("No capture spots for this platform; drew small markers.")

ax.set_title(f"{SLIDE_ID}: 500 \u00b5m molecular windows")
ax.set_xlim(0, image_width)
ax.set_ylim(image_height, 0)
ax.set_xlabel("Downscaled image x")
ax.set_ylabel("Downscaled image y")

FIGURE_DIR = OUTPUT_ATHENA_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
figure_path = FIGURE_DIR / f"{SLIDE_ID}_windows_overlay.png"

plt.tight_layout()
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {figure_path}")
