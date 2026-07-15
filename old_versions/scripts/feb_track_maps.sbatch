#!/bin/bash
#SBATCH --job-name=feb_track_maps
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_track_maps-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_track_maps-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# After combine+filter completes: map random KEPT vs would-DROP tracks on fixed data.

echo "Job started: $(date)"
PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
cd "$SUBROOT" || exit 1

$PYTHON -u scripts/map_filtered_tracks.py \
  --mode fixed-filtered \
  --filtered data/processed/combined_filtered/test.parquet \
  --n 16 \
  --seed 7 \
  --out-dir "data/results/USA Combined/unknown/track_maps"

# Also re-emit segments map for convenience
$PYTHON -u scripts/map_filtered_tracks.py \
  --mode segments \
  --n 8 \
  --seed 7 \
  --out-dir "data/results/USA Combined/unknown/track_maps"

echo "Job ended: $(date)"
