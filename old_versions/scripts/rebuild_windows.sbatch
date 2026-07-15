#!/bin/bash
#SBATCH --job-name=ais_rebuild_windows
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/rebuild_windows-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/rebuild_windows-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $SLURM_SUBMIT_DIR"

PYTHON=/home/projects/crml-prj10844/miniforge3/bin/python3
PROJECT=/home/projects/crml-prj10844/deep_learning/project/proj/project

cd "$PROJECT" || { echo "ERROR: project dir not found"; exit 1; }

echo "Python: $($PYTHON --version)"
echo "---"

$PYTHON scripts/rebuild_windows.py \
    --data-root data/processed \
    --all

echo "---"
echo "Job ended: $(date)"
