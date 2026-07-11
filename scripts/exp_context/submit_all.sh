#!/bin/bash
# Context-length suite from project_research.md
# Fixed 6h future; variable history: 3h, 6h, 12h, 24h LSTM + 24h Transformer
# Usage: bash submit_all.sh [afterok_job_id]
set -euo pipefail
SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_context
DEP="${1:-}"
DEP_ARG=()
if [ -n "$DEP" ]; then
  DEP_ARG=(--dependency=afterok:"$DEP")
  echo "Chaining after job $DEP"
fi

echo "=== exp_context serial: L3 -> L6 -> L12 -> L24 -> T24 (all: *h history -> 6h future) ==="
J1=$(sbatch --parsable "${DEP_ARG[@]}" "$SCRIPTS/train_L3_lstm.sbatch"); echo "L3 LSTM:        $J1"
J2=$(sbatch --parsable --dependency=afterok:$J1 "$SCRIPTS/train_L6_lstm.sbatch"); echo "L6 LSTM:        $J2"
J3=$(sbatch --parsable --dependency=afterok:$J2 "$SCRIPTS/train_L12_lstm.sbatch"); echo "L12 LSTM:       $J3"
J4=$(sbatch --parsable --dependency=afterok:$J3 "$SCRIPTS/train_L24_lstm.sbatch"); echo "L24 LSTM:       $J4"
J5=$(sbatch --parsable --dependency=afterok:$J4 "$SCRIPTS/train_T24_transformer.sbatch"); echo "T24 Transformer:$J5"

echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R' 2>/dev/null || true
