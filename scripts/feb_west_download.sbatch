#!/bin/bash
#SBATCH --job-name=feb_west
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_west_download-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_west_download-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# West coast: no day parquets yet. Download+clean all 28 Feb days, then finalize.
# Download step now retries on network errors and skips days that fail entirely.

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"

cd "$SUBROOT" || { echo "ERROR: subroot not found"; exit 1; }

START=2025-02-01
END=2025-02-28

echo ""
echo "=== West Coast: download+clean $START → $END ==="
$PYTHON "processing/INCREMENTAL_PROCESS West Coast.py" \
    --start-date "$START" \
    --end-date "$END" \
    --dataset-tag feb \
    --region west_coast \
    --no-finalize

echo ""
echo "=== West Coast: finalize ==="
$PYTHON "processing/INCREMENTAL_PROCESS West Coast.py" \
    --finalize-only \
    --dataset-tag feb \
    --region west_coast

echo ""
echo "Job ended: $(date)"
