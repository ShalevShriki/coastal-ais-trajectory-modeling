#!/bin/bash
# Experiment E: recursive 1-hour sliding-window forecaster
#SBATCH --job-name=ar_exp_E
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ar_exp_E-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ar_exp_E-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

set -euo pipefail
echo "Job started: $(date)"
echo "Experiment E: recursive 1h sliding window"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
cd "$SUBROOT"

$PYTHON -u models/RNN_recursive_1h.py \
    --coast "USA Combined" \
    --input data/processed/combined_filtered_smart/train.parquet \
    --run-tag "ar_exp/E" \
    --sample 400000 \
    --hidden-dim 256 --num-layers 2 --dropout 0.2 \
    --batch-size 256 --lr 1e-3 \
    --epochs 60 --patience 10 --no-maneuver-oversample --horizon-hours 12.0 \
    --residual-naive

echo "Job ended: $(date)"
