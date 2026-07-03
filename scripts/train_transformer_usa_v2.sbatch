#!/bin/bash
#SBATCH --job-name=ais_transformer_v2
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_transformer_usa_v2-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_transformer_usa_v2-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project

cd "$SUBROOT" || { echo "ERROR: subroot not found"; exit 1; }

echo "Python: $PYTHON"
echo "CUDA: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")')"

PYTHONPATH="$PROJECT" $PYTHON -u models/transformers.py \
    --coast "USA Combined" \
    --input data/processed/combined_filtered/train.parquet \
    --sample 1500000 \
    --d-model 128 \
    --nhead 8 \
    --num-encoder-layers 4 \
    --dim-feedforward 512 \
    --dropout 0.1 \
    --batch-size 128 \
    --epochs 60 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --patience 10 \
    --no-maneuver-oversample \
    --motion-balanced-sample \
    --straight-fraction 0.15 \
    --other-fraction 0.15 \
    --residual-naive \
    --horizon-hours 12.0

echo "Job ended: $(date)"
