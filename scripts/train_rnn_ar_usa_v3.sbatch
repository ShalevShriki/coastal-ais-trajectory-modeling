#!/bin/bash
#SBATCH --job-name=ais_rnn_ar_v3
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_rnn_ar_usa_v3-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_rnn_ar_usa_v3-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# AR 500k: ~2-4 min/epoch alone (autoregressive decode), expect ~30-40 epochs.
# Budget: ~20min load + ~1.5-2.5h train + ~20min eval/save ≈ 2-3h → 4h wall with buffer.

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"

cd "$SUBROOT" || exit 1

echo "Python: $PYTHON"
echo "CUDA: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")')"

PYTHONPATH="$PROJECT" $PYTHON -u models/RNN_AR.py \
    --coast "USA Combined" \
    --input data/processed/combined_filtered/train.parquet \
    --sample 500000 \
    --rnn-type lstm \
    --hidden-dim 256 \
    --num-layers 2 \
    --dropout 0.2 \
    --batch-size 256 \
    --epochs 60 \
    --lr 1e-3 \
    --patience 10 \
    --no-maneuver-oversample \
    --motion-balanced-sample \
    --straight-fraction 0.15 \
    --other-fraction 0.15 \
    --residual-naive \
    --horizon-hours 12.0 \
    --teacher-forcing 0.3

echo "Job ended: $(date)"
