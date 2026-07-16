#!/usr/bin/env bash
# Exact commands used for the report suite (exp_coastal).
# Source of truth mirrors scripts/exp_coastal/train_*.sbatch with shared defaults expanded.
#
# Usage (from proj/project, after download_processed_data.py):
#   export PYTHONPATH=.../parent_of_proj
#   bash scripts/exp_coastal/reproduce_experiments.sh           # print only
#   bash scripts/exp_coastal/reproduce_experiments.sh --run flat # run one
#   bash scripts/exp_coastal/reproduce_experiments.sh --run all  # run full suite (GPU, long)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATA="${DATA:-data/processed/combined_filtered_smart_coastal/train.parquet}"
SAMPLE="${SAMPLE:-300000}"
RUN_PREFIX="${RUN_PREFIX:-exp_coastal}"
FUTURE_H="${FUTURE_H:-12}"
HORIZON_H="${HORIZON_H:-12}"
LAND_PENALTY="${LAND_PENALTY:-0.1}"

SHARED_AR=(
  --coast "USA Combined" --input "$DATA"
  --sample "$SAMPLE" --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H"
  --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2
  --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10
  --no-maneuver-oversample --target-mode anchor_offset
  --land-penalty-weight "$LAND_PENALTY"
)

cmd_ar9()   { echo "$PYTHON" -u models/RNN_AR.py --run-tag "$RUN_PREFIX/AR_9h"  --history-hours 9  "${SHARED_AR[@]}"; }
cmd_ar12()  { echo "$PYTHON" -u models/RNN_AR.py --run-tag "$RUN_PREFIX/AR_12h" --history-hours 12 "${SHARED_AR[@]}"; }
cmd_ar18()  { echo "$PYTHON" -u models/RNN_AR.py --run-tag "$RUN_PREFIX/AR_18h" --history-hours 18 "${SHARED_AR[@]}"; }
cmd_ar24()  { echo "$PYTHON" -u models/RNN_AR.py --run-tag "$RUN_PREFIX/AR_24h" --history-hours 24 "${SHARED_AR[@]}"; }
cmd_ar12_noland() {
  echo "$PYTHON" -u models/RNN_AR.py --run-tag "$RUN_PREFIX/AR_12h_noland" --history-hours 12 \
    --coast "USA Combined" --input "$DATA" \
    --sample "$SAMPLE" --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
    --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
    --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
    --no-maneuver-oversample --target-mode anchor_offset \
    --land-penalty-weight 0.0
}
cmd_flat() {
  echo "$PYTHON" -u models/RNN.py --run-tag "$RUN_PREFIX/flat_lstm" \
    --coast "USA Combined" --input "$DATA" \
    --sample "$SAMPLE" --history-hours 24 --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
    --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
    --batch-size 256 --lr 1e-3 --epochs 60 --patience 10 \
    --no-maneuver-oversample --no-curriculum \
    --land-penalty-weight "$LAND_PENALTY"
}
cmd_transformer() {
  echo "$PYTHON" -u models/transformers.py --run-tag "$RUN_PREFIX/transformer" \
    --coast "USA Combined" --input "$DATA" \
    --sample "$SAMPLE" --history-hours 24 --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
    --d-model 128 --nhead 8 --num-encoder-layers 4 --dim-feedforward 512 --dropout 0.1 \
    --batch-size 128 --lr 1e-3 --weight-decay 1e-4 --epochs 60 --patience 10 \
    --no-maneuver-oversample --no-curriculum \
    --land-penalty-weight "$LAND_PENALTY"
}
cmd_adaptive() {
  echo "$PYTHON" -u models/RNN_AR_adaptive.py --run-tag "$RUN_PREFIX/adaptive_multiscale" \
    --coast "USA Combined" --input "$DATA" \
    --sample "$SAMPLE" --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
    --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
    --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
    --no-maneuver-oversample \
    --land-penalty-weight "$LAND_PENALTY"
}
cmd_sliding() {
  echo "$PYTHON" -u models/RNN_recursive_1h.py --run-tag "$RUN_PREFIX/sliding_3h" \
    --coast "USA Combined" --input "$DATA" \
    --sample "$SAMPLE" --chunk-hours 3 --horizon-hours "$HORIZON_H" \
    --rnn-type lstm --hidden-dim 256 --num-layers 2 --dropout 0.2 \
    --batch-size 256 --lr 1e-3 --epochs 60 --patience 10 \
    --no-maneuver-oversample --no-curriculum \
    --land-penalty-weight "$LAND_PENALTY"
}
cmd_sep_hard() {
  echo "$PYTHON" -u models/RNN_AR_diff_encoder.py --run-tag "$RUN_PREFIX/adaptive_separate_encoders_hard" \
    --gate-mode hard \
    --coast "USA Combined" --input "$DATA" \
    --sample "$SAMPLE" --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
    --hidden-dim 256 --num-layers 2 --dropout 0.2 \
    --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
    --no-maneuver-oversample \
    --land-penalty-weight "$LAND_PENALTY"
}
cmd_sep_softmax() {
  echo "$PYTHON" -u models/RNN_AR_diff_encoder.py --run-tag "$RUN_PREFIX/adaptive_separate_encoders_softmax" \
    --gate-mode softmax \
    --coast "USA Combined" --input "$DATA" \
    --sample "$SAMPLE" --future-hours "$FUTURE_H" --horizon-hours "$HORIZON_H" \
    --hidden-dim 256 --num-layers 2 --dropout 0.2 \
    --batch-size 256 --lr 1e-3 --teacher-forcing 0.3 --epochs 60 --patience 10 \
    --no-maneuver-oversample \
    --land-penalty-weight "$LAND_PENALTY"
}

print_all() {
  echo "# Shared defaults: SAMPLE=$SAMPLE FUTURE_H=$FUTURE_H HORIZON_H=$HORIZON_H LAND_PENALTY=$LAND_PENALTY"
  echo "# DATA=$DATA"
  echo
  for name in ar9 ar12 ar18 ar24 ar12_noland flat transformer adaptive sliding sep_hard sep_softmax; do
    echo "### $name"
    "cmd_$name"
    echo
  done
}

run_one() {
  local name="$1"
  local line
  line=$("cmd_$name")
  echo "+ $line"
  eval "$line"
}

MODE="print"
TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) MODE="run"; TARGET="${2:-}"; shift 2 ;;
    --print) MODE="print"; shift ;;
    -h|--help)
      echo "Usage: $0 [--print] | --run {ar9|ar12|ar18|ar24|ar12_noland|flat|transformer|adaptive|sliding|sep_hard|sep_softmax|all}"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$MODE" == "print" ]]; then
  if [[ ! -f "$DATA" ]]; then
    echo "# Note: $DATA not found yet — run: python scripts/download_processed_data.py" >&2
  fi
  print_all
  exit 0
fi

if [[ ! -f "$DATA" ]]; then
  echo "Missing $DATA — run: python scripts/download_processed_data.py" >&2
  exit 1
fi

case "$TARGET" in
  all)
    for name in ar9 ar12 ar18 ar24 flat transformer adaptive sliding; do
      run_one "$name"
    done
    ;;
  ar9|ar12|ar18|ar24|ar12_noland|flat|transformer|adaptive|sliding|sep_hard|sep_softmax)
    run_one "$TARGET"
    ;;
  *)
    echo "Unknown experiment: $TARGET" >&2
    exit 1
    ;;
esac
