"""
Paths shared by all scripts.

Input data are read from DATA_ROOT, which defaults to ``data/`` at the
repository root and can be set with the DATA_ROOT environment variable.
Expected layout:

    DATA_ROOT/breast/wsis/<SLIDE>.tif                  whole-slide images
    DATA_ROOT/breast/st/<SLIDE>.h5ad                   spatial transcriptomics (HEST-1k)
    DATA_ROOT/athena/{cell_df,cell_graphs,spot_df,window_df,features_df}/
    DATA_ROOT/athena_2/<SLIDE>_athena_features.csv     per-window ATHENA features

Results are written to the output/ folder of each stage.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))

# Inputs
BREAST_WSI_DIR = DATA_ROOT / "breast" / "wsis"
BREAST_ST_DIR = DATA_ROOT / "breast" / "st"
ATHENA_DIR = DATA_ROOT / "athena"
ATHENA2_DIR = DATA_ROOT / "athena_2"

# Outputs
CELLVIT_OUT = REPO_ROOT / "1_cellvit" / "output" / "cellvit_out"
QUPATH_OUT = REPO_ROOT / "1_cellvit" / "output" / "qupath_export"
ATHENA_OUT = REPO_ROOT / "2_athena" / "output"
STATE_OUT = REPO_ROOT / "3_spatial_state" / "output"
STEP5_TABLES = STATE_OUT / "step5_interaction_state" / "tables"
FIGURES_OUT = STATE_OUT / "manuscript_figures"

# Slides left out of the cohort (101 slides remain)
EXCLUDED_SLIDES = {
    **{f"SPA{n}": "duplicate slide" for n in range(119, 131)},
    "TENX94": "incorrect pixel size",
    "TENX95": "incorrect pixel size",
}
