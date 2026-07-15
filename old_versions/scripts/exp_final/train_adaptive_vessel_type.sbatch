#!/bin/bash
#SBATCH --job-name=ef_adapt_vt
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_final_adaptive_vt-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_final_adaptive_vt-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_final/_env.sh
cd "$SUBROOT"
echo "Adaptive multi-scale AR + vessel-type gate | started $(date)"
echo "Run tag: $RUN_PREFIX/adaptive_vessel_type (does not overwrite adaptive_multiscale)"
$PYTHON -u models/RNN_AR_adaptive.py --coast "USA Combined" --input "$DATA" \
  --run-tag "$RUN_PREFIX/adaptive_vessel_type" \
  --gate-vessel-type \
  --sample "$SAMPLE" --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
  --no-maneuver-oversample
echo "ended $(date)"
