#!/bin/bash
#SBATCH --job-name=feb_mex_fin
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_mexican_finalize-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_mexican_finalize-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# Mexican coast: all 28 Feb day parquets are already on disk.
# Only the window-build step OOM'd previously; the builder now streams to
# parquet in batches, so finalize-only should fit comfortably in 192G.

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"

cd "$SUBROOT" || { echo "ERROR: subroot not found"; exit 1; }

echo ""
echo "=== Mexican Coast: finalize only (28 day parquets on disk) ==="
$PYTHON "processing/INCREMENTAL_PROCESS Mexcany Beach.py" \
    --finalize-only \
    --dataset-tag feb \
    --region mexican_coast

echo ""
echo "Job ended: $(date)"
