#!/bin/bash
#SBATCH --job-name=e1v2_xf
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp1_v2_residual_transformer-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp1_v2_residual_transformer-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "v2/experiment1 / transformer"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
cd "$SUBROOT" || exit 1

echo "CUDA: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO CUDA\")')"

PYTHONPATH="$PROJECT" $PYTHON -u models/transformers.py \
    --coast "USA Combined" \
    --input data/processed/combined_filtered/train.parquet \
    --run-tag "v2/experiment1" \
    --sample 400000 \
    --d-model 128 --nhead 8 --num-encoder-layers 4 --dim-feedforward 512 --dropout 0.1 --batch-size 128 --lr 1e-3 --weight-decay 1e-4 \
    --epochs 60 \
    --patience 10 \
    --no-maneuver-oversample \
    --horizon-hours 12.0 \
    --motion-balanced-sample --straight-fraction 0.15 --other-fraction 0.15 --residual-naive

echo "Job ended: $(date)"
