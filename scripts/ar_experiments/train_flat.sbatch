#!/bin/bash
# Flat RNN baseline for Comparison 1 (direct multi-step)
#SBATCH --job-name=ar_exp_flat
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ar_exp_flat-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ar_exp_flat-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

set -euo pipefail
echo "Job started: $(date)"
echo "Baseline: Flat LSTM (direct multi-step)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
cd "$SUBROOT"

$PYTHON -u models/RNN.py \
    --coast "USA Combined" \
    --input data/processed/combined_filtered_smart/train.parquet \
    --run-tag "ar_exp/flat" \
    --sample 400000 \
    --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
    --batch-size 256 --lr 1e-3 \
    --epochs 60 --patience 10 --no-maneuver-oversample --horizon-hours 12.0

echo "Job ended: $(date)"
