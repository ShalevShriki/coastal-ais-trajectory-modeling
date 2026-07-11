#!/bin/bash
#SBATCH --job-name=ec_M3_sw
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_clean_M3-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_clean_M3-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_clean/_env.sh
cd "$SUBROOT"
echo "M3 sliding 3h + residual | started $(date)"
$PYTHON -u models/RNN_recursive_1h.py --coast "USA Combined" --input "$DATA" --run-tag "$RUN_PREFIX/M3_ar_sliding_3h_res" \
  --sample "$SAMPLE" --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --epochs 60 --patience 10 --no-maneuver-oversample \
  --horizon-hours 12.0 --chunk-hours 3 --residual-naive
echo "ended $(date)"
