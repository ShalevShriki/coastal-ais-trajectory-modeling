#!/bin/bash
#SBATCH --job-name=ec_adapt_hard
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_coastal_adaptive_sep_hard-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_coastal_adaptive_sep_hard-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_coastal/_env.sh
cd "$SUBROOT"
mkdir -p LOG
echo "Coastal diff-encoder adaptive (hard Gumbel gate) | started $(date)"
$PYTHON -u models/RNN_AR_diff_encoder.py --coast "USA Combined" --input "$DATA" \
  --run-tag "$RUN_PREFIX/adaptive_separate_encoders_hard" \
  --gate-mode hard \
  --sample "$SAMPLE" --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
  --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
  --no-maneuver-oversample \
  --land-penalty-weight "$LAND_PENALTY"
echo "ended $(date)"
