#!/bin/bash
#SBATCH --job-name=ais_all_coasts_14d
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/all_coasts_14d-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/all_coasts_14d-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# Download and process 14 days of NOAA AIS data for ALL three coastal regions.
# Date range: 2025-03-01 to 2025-03-14 (matches existing East Coast 14d dataset).
# Each region runs sequentially to avoid download conflicts on shared temp dir.
# After all three are done, the combine script merges them into train/val/test.

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/bin/python3
PROJECT=/home/projects/crml-prj10844/deep_learning/project/proj/project

cd "$PROJECT" || { echo "ERROR: project dir not found"; exit 1; }

START=2025-03-01
END=2025-03-14
TAG=14d

echo ""
echo "====================================================="
echo " Processing all 3 coasts: $START → $END (tag=$TAG)"
echo "====================================================="

# ---- 1. East Coast ----------------------------------------------------------
echo ""
echo "=== [1/3] East Coast ==="
$PYTHON "processing/INCREMENTAL_PROCESS Eastern coast.py" \
    --start-date "$START" \
    --end-date   "$END"   \
    --dataset-tag "$TAG"  \
    --region east_coast
echo "East Coast done at: $(date)"

# ---- 2. Mexican Coast -------------------------------------------------------
echo ""
echo "=== [2/3] Mexican Coast ==="
$PYTHON "processing/INCREMENTAL_PROCESS Mexcany Beach.py" \
    --start-date "$START" \
    --end-date   "$END"   \
    --dataset-tag "$TAG"  \
    --region mexican_coast
echo "Mexican Coast done at: $(date)"

# ---- 3. West Coast ----------------------------------------------------------
echo ""
echo "=== [3/3] West Coast ==="
$PYTHON "processing/INCREMENTAL_PROCESS West Coast.py" \
    --start-date "$START" \
    --end-date   "$END"   \
    --dataset-tag "$TAG"  \
    --region west_coast
echo "West Coast done at: $(date)"

# ---- Combine into one dataset -----------------------------------------------
echo ""
echo "=== Combining all coasts into train/val/test ==="
$PYTHON scripts/combine_datasets.py --include-misplaced
echo "Combine done at: $(date)"

echo ""
echo "Job ended: $(date)"
