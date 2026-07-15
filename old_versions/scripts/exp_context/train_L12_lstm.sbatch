#!/bin/bash
#SBATCH --job-name=ctx_L12
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_context_L12-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_context_L12-%j.out
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_context/_env.sh
cd "$SUBROOT"
echo "LSTM 12h history -> 6h future | started $(date)"
$PYTHON -u models/RNN.py --coast "USA Combined" --input "$DATA" --run-tag "$RUN_PREFIX/L12_lstm_12h" \
  --sample "$SAMPLE" --history-hours 12 --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
  --batch-size 256 --lr 1e-3 --epochs 60 --patience 10 --no-maneuver-oversample --no-curriculum
echo "ended $(date)"
