#!/bin/bash
#SBATCH --job-name=feb_filter
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_filter_only-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_filter_only-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"

cd "$SUBROOT" || exit 1

$PYTHON scripts/apply_training_filters.py

echo "Job ended: $(date)"
