#!/bin/bash
#SBATCH --job-name=e1v1_ar
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp1_v1_baseline_rnn_ar-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp1_v1_baseline_rnn_ar-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "v1/experiment1 / rnn_ar"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
cd "$SUBROOT" || exit 1

echo "CUDA: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO CUDA\")')"

PYTHONPATH="$PROJECT" $PYTHON -u models/RNN_AR.py \
    --coast "USA Combined" \
    --input data/processed/combined_filtered/train.parquet \
    --run-tag "v1/experiment1" \
    --sample 400000 \
    --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 \
    --epochs 60 \
    --patience 10 \
    --no-maneuver-oversample \
    --horizon-hours 12.0 \
    

echo "Job ended: $(date)"
