#!/bin/bash
#SBATCH --job-name=feb_combine
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_combine_filter-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_combine_filter-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# Combine the three Feb coast window sets into data/processed/combined,
# then apply the stationary/motion filters to produce combined_filtered.

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"

cd "$SUBROOT" || { echo "ERROR: subroot not found"; exit 1; }

echo ""
echo "=== Combine February-only -> data/processed/combined ==="
$PYTHON scripts/combine_datasets.py --include-misplaced --tag feb
echo "Combine done at: $(date)"

echo ""
echo "=== Stationary filter -> combined_filtered ==="
$PYTHON scripts/apply_training_filters.py
echo "Filter done at: $(date)"

echo ""
echo "Job ended: $(date)"
