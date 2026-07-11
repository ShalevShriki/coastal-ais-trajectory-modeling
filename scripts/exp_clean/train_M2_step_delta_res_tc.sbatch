#!/bin/bash
#SBATCH --job-name=ec_M2_ar
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_clean_M2-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_clean_M2-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_clean/_env.sh
cd "$SUBROOT"
echo "M2 AR step-delta + residual + TC | started $(date)"
$PYTHON -u models/RNN_AR.py --coast "USA Combined" --input "$DATA" --run-tag "$RUN_PREFIX/M2_ar_step_delta_res_tc" \
  --sample "$SAMPLE" --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
  --no-maneuver-oversample --horizon-hours 12.0 \
  --residual-naive --target-mode step_delta
echo "ended $(date)"
