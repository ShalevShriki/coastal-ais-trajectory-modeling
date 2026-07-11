#!/bin/bash
#SBATCH --job-name=ec_B2_xf
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_clean_B2-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_clean_B2-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_clean/_env.sh
cd "$SUBROOT"
echo "B2 Transformer | started $(date)"
$PYTHON -u models/transformers.py --coast "USA Combined" --input "$DATA" --run-tag "$RUN_PREFIX/B2_transformer" \
  --sample "$SAMPLE" --d-model 128 --nhead 8 --num-encoder-layers 4 --dim-feedforward 512 --dropout 0.1 \
  --batch-size 128 --lr 1e-3 --weight-decay 1e-4 --epochs 60 --patience 10 --no-maneuver-oversample \
  --horizon-hours 12.0 --no-curriculum
echo "ended $(date)"
