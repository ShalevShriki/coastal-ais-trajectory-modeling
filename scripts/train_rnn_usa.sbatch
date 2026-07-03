#!/bin/bash
#SBATCH --job-name=ais_rnn_usa
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_rnn_usa-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/ais_rnn_usa-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

PYTHONPATH="$PROJECT" $PYTHON -u models/RNN.py \
    --coast "USA Combined" \
    --input data/processed/combined_filtered/train.parquet \
    --sample 400000 \
    --rnn-type lstm \
    --hidden-dim 256 \
    --num-layers 2 \
    --dropout 0.2 \
    --batch-size 256 \
    --epochs 60 \
    --lr 1e-3 \
    --patience 10 \
    --no-maneuver-oversample \
    --horizon-hours 12.0

echo "Job ended: $(date)"
