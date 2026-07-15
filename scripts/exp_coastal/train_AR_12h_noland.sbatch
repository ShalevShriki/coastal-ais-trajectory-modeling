#!/bin/bash
#SBATCH --job-name=ec_AR12_nlp
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_coastal_AR12_noland-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_coastal_AR12_noland-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_coastal/_env.sh
cd "$SUBROOT"
echo "Ablation: Coastal AR 12h, land_penalty=0 | started $(date)"
echo "Run tag: $RUN_PREFIX/AR_12h_noland (does not overwrite AR_12h)"
$PYTHON -u models/RNN_AR.py --coast "USA Combined" --input "$DATA" \
  --run-tag "$RUN_PREFIX/AR_12h_noland" \
  --sample "$SAMPLE" --history-hours 12 --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
  --no-maneuver-oversample --target-mode anchor_offset \
  --land-penalty-weight 0.0
echo "ended $(date)"
