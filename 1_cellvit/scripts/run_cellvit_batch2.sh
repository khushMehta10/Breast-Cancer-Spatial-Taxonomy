#!/bin/bash
#SBATCH --job-name=cellvit_breast
#SBATCH --output=logs/cellvit_%A_%a.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=b40x4
#SBATCH --array=100-123%3

# Submit from the 1_cellvit/ directory:  mkdir -p logs && sbatch scripts/run_cellvit_batch2.sh
# Input slides are read from $DATA_ROOT/breast/wsis (default: ../data).

module load miniconda/3          # adjust for your system
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellvit
DATA_ROOT="${DATA_ROOT:-$SLURM_SUBMIT_DIR/../data}"
WSI_DIR="$DATA_ROOT/breast/wsis"
OUT="$SLURM_SUBMIT_DIR/output/cellvit_out"
mkdir -p "$OUT"
mapfile -t WSIS < <(ls "$WSI_DIR"/*.tif | sort)
WSI="${WSIS[$SLURM_ARRAY_TASK_ID]}"
SID=$(basename "$WSI" .tif)
echo "start $SID"
MPP=$(python -c "import openslide; s=openslide.OpenSlide('$WSI'); print(s.properties.get('openslide.mpp-x','NONE'))")
cellvit-inference --model HIPT --nuclei_taxonomy pannuke --outdir "$OUT" --geojson --cpu_count 4 --ray_worker 1 --ray_remote_cpus 1 process_wsi --wsi_path "$WSI" --wsi_mpp "$MPP" --wsi_magnification 40
echo "done $SID"
