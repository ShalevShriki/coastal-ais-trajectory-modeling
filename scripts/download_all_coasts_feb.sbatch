#!/bin/bash
#SBATCH --job-name=ais_feb_2025
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_feb_2025-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_feb_2025-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# Full February 2025 NOAA AIS for all three US coastal regions.
# Stored under dataset tag "feb" (separate from March 14d data).
# Combines ONLY feb-tagged windows into data/processed/combined (overwrites).

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project

cd "$SUBROOT" || { echo "ERROR: subroot not found"; exit 1; }

START=2025-02-01
END=2025-02-28
TAG=feb

run_coast() {
    local name="$1"
    local script="$2"
    local region="$3"
    echo ""
    echo "=== $name: $START → $END (tag=$TAG) ==="
    $PYTHON "$script" \
        --start-date "$START" \
        --end-date "$END" \
        --dataset-tag "$TAG" \
        --region "$region"
    echo "=== $name: finalize ==="
    $PYTHON "$script" \
        --finalize-only \
        --dataset-tag "$TAG" \
        --region "$region"
    echo "$name done at: $(date)"
}

echo ""
echo "====================================================="
echo " February 2025 — all 3 coasts ($START → $END)"
echo "====================================================="

run_coast "East Coast"   "processing/INCREMENTAL_PROCESS Eastern coast.py" "east_coast"
run_coast "Mexican Coast" "processing/INCREMENTAL_PROCESS Mexcany Beach.py" "mexican_coast"
run_coast "West Coast"   "processing/INCREMENTAL_PROCESS West Coast.py"   "west_coast"

echo ""
echo "=== Combine February-only -> data/processed/combined ==="
$PYTHON scripts/combine_datasets.py --include-misplaced --tag "$TAG"
echo "Combine done at: $(date)"

echo ""
echo "=== Stationary filter -> combined_filtered ==="
$PYTHON scripts/apply_training_filters.py
echo "Filter done at: $(date)"

echo ""
echo "Job ended: $(date)"
