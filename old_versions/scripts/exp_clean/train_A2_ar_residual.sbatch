#!/bin/bash
#SBATCH --job-name=ec_A2_ar
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_clean_A2-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_clean_A2-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_clean/_env.sh
cd "$SUBROOT"
echo "A2 AR anchor + residual, no TC | started $(date)"
$PYTHON -u models/RNN_AR.py --coast "USA Combined" --input "$DATA" --run-tag "$RUN_PREFIX/A2_ar_anchor_residual" \
  --sample "$SAMPLE" --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
  --no-maneuver-oversample --horizon-hours 12.0 \
  --residual-naive --no-curriculum --no-scheduled-tf --target-mode anchor_offset
echo "ended $(date)"
