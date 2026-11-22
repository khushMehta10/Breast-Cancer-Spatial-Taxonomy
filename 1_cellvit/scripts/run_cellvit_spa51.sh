#!/bin/bash
#SBATCH --job-name=cellvit_spa51
#SBATCH --output=logs/cellvit_spa51_%j.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=b40x4

# Submit from the 1_cellvit/ directory:  mkdir -p logs && sbatch scripts/run_cellvit_spa51.sh
# Input slides are read from $DATA_ROOT/breast/wsis (default: ../data).

module load miniconda/3          # adjust for your system
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellvit
DATA_ROOT="${DATA_ROOT:-$SLURM_SUBMIT_DIR/../data}"
WSI="$DATA_ROOT/breast/wsis/SPA51.tif"
OUT="$SLURM_SUBMIT_DIR/output/cellvit_out"
mkdir -p "$OUT"

cellvit-inference --model HIPT --nuclei_taxonomy pannuke \
    --outdir $OUT --geojson --cpu_count 4 --ray_worker 1 --ray_remote_cpus 1 \
    process_wsi --wsi_path $WSI --wsi_mpp 0.6861 --wsi_magnification 20

echo "done -> $OUT"
