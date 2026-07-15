#!/bin/bash
#SBATCH --job-name=ec_xfmr
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_coastal_xfmr-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_coastal_xfmr-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_coastal/_env.sh
cd "$SUBROOT"
echo "Coastal Transformer 24h -> 12h | started $(date)"
$PYTHON -u models/transformers.py --coast "USA Combined" --input "$DATA" --run-tag "$RUN_PREFIX/transformer" \
  --sample "$SAMPLE" --history-hours 24 --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
  --d-model 128 --nhead 8 --num-encoder-layers 4 --dim-feedforward 512 --dropout 0.1 \
  --batch-size 128 --lr 1e-3 --weight-decay 1e-4 --epochs 60 --patience 10 \
  --no-maneuver-oversample --no-curriculum \
  --land-penalty-weight "$LAND_PENALTY"
echo "ended $(date)"
