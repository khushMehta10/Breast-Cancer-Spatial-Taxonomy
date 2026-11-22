# Breast Cancer Spatial Taxonomy

Tumour–immune spatial states in breast cancer H&E whole-slide images.

Nuclei are segmented and typed with CellViT, linked into a k = 5 nearest-neighbour
cell graph, and summarised per 500 × 500 µm window with ATHENA spatial features.
Each window that holds both immune and tumour cells is then tested against three
rule-based states:

- **distant co-occurrence**: both compartments present and clustered, interacting below chance
- **bordering**: immune and tumour in contact without penetration
- **infiltrated**: immune cells dispersed among tumour cells

Slides and spatial-transcriptomics data come from the HEST-1k collection
(sample IDs `SPA*`, `TENX*`, `NCBI*`).

```
config.py         input and output paths used by every script
envs/             conda environments
1_cellvit/        nuclei segmentation and typing (CellViT, PanNuke classes)
2_athena/         per-cell tables, kNN graphs, ATHENA feature figures
3_spatial_state/  window classification and state detection (the main pipeline)
```

## Setup

**Environments**

```
conda env create -f envs/cellvit.yml   # stage 1
conda env create -f envs/hest.yml      # stages 2-3
```

After creating the `cellvit` environment, apply the rounding patch described in
`1_cellvit/PATCH_NOTES.txt`.

**Data**

Input data are read from `data/` at the repository root. To keep them elsewhere,
set the `DATA_ROOT` environment variable. The expected layout is listed at the top
of `config.py`:

```
data/breast/wsis/<SLIDE>.tif                   whole-slide images
data/breast/st/<SLIDE>.h5ad                    spatial transcriptomics (HEST-1k)
data/athena/                                   per-cell and per-window ATHENA tables
data/athena_2/<SLIDE>_athena_features.csv      per-window ATHENA features
```

Results are written to the `output/` folder of each stage.

**Cohort**

101 breast slides. Fourteen further slides are excluded and listed in
`config.EXCLUDED_SLIDES`: 12 duplicate slides (SPA119–SPA130) and 2 with an
incorrect pixel size (TENX94, TENX95).

## Stage 1: CellViT (`1_cellvit/`)

| Path | Contents |
|---|---|
| `scripts/run_cellvit_batch1.sh` | SLURM array job, slides 0–99 (20× magnification) |
| `scripts/run_cellvit_batch2.sh` | SLURM array job, slides 100–123 (40×, pixel size read from the slide) |
| `scripts/run_cellvit_spa51.sh` | single-slide run (SPA51) |
| `scripts/cellvit_status.sh` | queue, progress and failure summary |
| `scripts/qupath/export_cells_qupath.py` | CellViT nuclei → QuPath overlays, coloured by cell type |
| `scripts/qupath/export_spots_qupath.py` | spatial-transcriptomics spots at their physical size |
| `scripts/qupath/export_center_boxes_qupath.py` | 512/1024/2048 px boxes around the central spot of selected slides |
| `scripts/qupath/export_spot_boxes_qupath.py` | a 1024 px box around every spot, all slides |
| `PATCH_NOTES.txt` | a required CellViT rounding patch, and the 6 slides excluded for mpp > 0.75 |

CellViT is run with the HIPT backbone and PanNuke taxonomy
(`cellvit-inference --model HIPT --nuclei_taxonomy pannuke`).

## Stage 2: ATHENA (`2_athena/`)

| Path | Contents |
|---|---|
| `scripts/tables/build_cell_tables.py` | CellViT output → one per-cell table per slide, with contour morphology |
| `scripts/tables/build_cell_table_one_slide.py` | the same for a single slide |
| `scripts/graphs/build_knn_graphs.py` | per-cell tables → k = 5 kNN adjacency matrices |
| `scripts/graphs/build_knn_graph_one_slide.py` | the same for a single slide |
| `scripts/figures/plot_infiltration_maps.py` | per-window infiltration scores over the tissue outline |
| `scripts/figures/plot_local_richness.py` | local phenotype richness per cell |
| `scripts/figures/plot_local_shannon.py` | local Shannon entropy per cell |
| `scripts/figures/plot_window_overlay.py` | 500 µm windows and spots over the H&E |
| `scripts/figures/plot_window_subgraph.py` | the cell graph inside one window |
| `scripts/notebooks/` | cell-table, kNN-graph and cell-interaction notebooks |

## Stage 3: Spatial state (`3_spatial_state/`)

Reads the per-window ATHENA features (`data/athena_2/`, computed with ATHENA from the
per-cell tables and kNN graphs) and runs, in order:

```
scripts/pipeline/classify_windows.py    step 1  window classes by immune / tumour presence
scripts/pipeline/measure_abundance.py   step 2  immune and tumour abundance per window
scripts/pipeline/spatial_state.py       step 3  per-phenotype arrangement (isolated, dispersed, aggregate, broad)
scripts/pipeline/interaction_state.py   step 5  tumour–immune state (distant, bordering, infiltrated)
scripts/pipeline/tme_engagement.py      step 6  engagement with connective tissue and epithelium, tumour and immune burden
```

There is no separate step 4; `spatial_state.py` covers steps 3–4.

| Path | Contents |
|---|---|
| `scripts/analysis/state_permutation_null.py` | permutation null for the three state definitions, and a check that they reproduce the pipeline labels |
| `scripts/analysis/state_composition.py` | median nucleus counts and compartment arrangement per state |
| `scripts/analysis/verify_results.py` | every number reported in the Results, taken from the pipeline output |
| `scripts/figures/fig_pipeline_overview.py` | pipeline overview figure |
| `scripts/figures/fig_state_overlay.py` | one window per state at nucleus resolution |
| `scripts/figures/fig_state_schematic.py` | schematic of the three states |
| `scripts/figures/plot_state_maps.py` | per-slide maps of the interaction states over the H&E |

The analysis scripts read the step 5 tables and print their results.

## Running

Run from the repository root unless noted.

```
# Stage 1 (SLURM, from 1_cellvit/)
cd 1_cellvit && mkdir -p logs && sbatch scripts/run_cellvit_batch1.sh && cd ..

conda activate hest

# Stage 2
python 2_athena/scripts/tables/build_cell_tables.py
python 2_athena/scripts/graphs/build_knn_graphs.py

# Stage 3
python 3_spatial_state/scripts/pipeline/classify_windows.py
python 3_spatial_state/scripts/pipeline/measure_abundance.py
python 3_spatial_state/scripts/pipeline/spatial_state.py
python 3_spatial_state/scripts/pipeline/interaction_state.py
python 3_spatial_state/scripts/pipeline/tme_engagement.py
```

The SLURM settings (partition, GPU, time limits) in the stage 1 scripts should be
adjusted for the cluster in use.

## References

Methods used: CellViT (Hörst et al., 2024) run with the CellViT++ inference package
(Hörst et al., 2025), ATHENA (Martinelli & Rapsomaniki, 2022), HEST-1k (Jaume et al., 2024).
