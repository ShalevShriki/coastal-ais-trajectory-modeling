#!/bin/bash
#SBATCH --job-name=ais_usa_wide_14d
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/usa_wide_14d-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/usa_wide_14d-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# Single wide-bbox AIS download — captures cross-coast long trajectories.
#
# Instead of tight per-region boxes that cut trajectories at regional borders,
# we use two large boxes matching connected bodies of water:
#
#   usa_atlantic  lat 24–47  lon -100 to -60   (East Coast + Gulf of Mexico)
#   usa_pacific   lat 14–50  lon -130 to -105  (Pacific coast: Mexico to Canada)
#
# A ship going Boston → Miami now appears as ONE continuous trajectory instead
# of being cut at the East Coast southern bbox boundary (lat 35).
#
# Date range: 2025-03-01 to 2025-03-14 (same 14-day window as other datasets).

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/bin/python3
PROJECT=/home/projects/crml-prj10844/deep_learning/project/proj/project

cd "$PROJECT" || { echo "ERROR: project dir not found"; exit 1; }

START=2025-03-01
END=2025-03-14

echo ""
echo "============================================================"
echo " Wide-bbox USA AIS — $START to $END"
echo "============================================================"

# ---- 1. Atlantic + Gulf of Mexico ------------------------------------------
echo ""
echo "=== [1/2] Atlantic + Gulf (lat 24-47, lon -100 to -60) ==="
$PYTHON "processing/INCREMENTAL_PROCESS Eastern coast.py" \
    --start-date "$START" \
    --end-date   "$END"   \
    --dataset-tag usa_atlantic \
    --lat-min 24.0 --lat-max 47.0 \
    --lon-min -100.0 --lon-max -60.0 \
    --no-windows
echo "Atlantic+Gulf segments done at: $(date)"

# ---- 2. Pacific Coast (Mexico to Canada) ------------------------------------
echo ""
echo "=== [2/2] Pacific coast (lat 14-50, lon -130 to -105) ==="
$PYTHON "processing/INCREMENTAL_PROCESS West Coast.py" \
    --start-date "$START" \
    --end-date   "$END"   \
    --dataset-tag usa_pacific \
    --lat-min 14.0 --lat-max 50.0 \
    --lon-min -130.0 --lon-max -105.0 \
    --no-windows
echo "Pacific segments done at: $(date)"

# ---- Build windows with memory-efficient numpy builder ----------------------
echo ""
echo "=== Building windows (memory-efficient) ==="
$PYTHON scripts/rebuild_windows.py \
    --segments "data/processed/Eastern coast/ais_custom_usa_atlantic_long_horizon/coastal_segments.parquet"
$PYTHON scripts/rebuild_windows.py \
    --segments "data/processed/West Coast/ais_custom_usa_pacific_long_horizon/coastal_segments.parquet"

# ---- Combine ----------------------------------------------------------------
echo ""
echo "=== Combining into train/val/test ==="
$PYTHON scripts/combine_datasets.py --include-misplaced
echo "Combine done at: $(date)"

echo ""
echo "Job ended: $(date)"
