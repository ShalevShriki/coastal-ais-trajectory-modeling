#!/bin/bash
#SBATCH --job-name=ais_transformer_v3
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_transformer_usa_v3-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_transformer_usa_v3-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# Transformer 600k: ~153s/epoch alone, expect ~45-55 epochs (patience 10).
# Budget: ~15min load + ~2.5h train + ~20min eval/save ≈ 3h → 4h wall with buffer.

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"

cd "$SUBROOT" || exit 1

echo "Python: $PYTHON"
echo "CUDA: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")')"

PYTHONPATH="$PROJECT" $PYTHON -u models/transformers.py \
    --coast "USA Combined" \
    --input data/processed/combined_filtered/train.parquet \
    --sample 600000 \
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
