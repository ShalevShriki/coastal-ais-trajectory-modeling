#!/bin/bash
# Usage: ./submit.sh [FILTER_JOB_ID]
# 1) smart filter (if no job id given, submits filter first)
# 2) v1 + v1_residual on combined_filtered_smart (6 GPU jobs parallel)
set -euo pipefail

ROOT=/home/projects/crml-prj10844/deep_learning/project/proj/project
SCRIPTS="$ROOT/scripts/experiment1/smart_motion"

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  FILTER_JOB="$1"
else
  FILTER_JOB=$(sbatch --parsable "$ROOT/scripts/smart_motion_filter.sbatch")
  echo "smart filter job: $FILTER_JOB"
fi

DEP="--dependency=afterok:${FILTER_JOB}"

submit() {
  sbatch --parsable $DEP "$1"
}

echo "=== v1/smart_motion (after filter $FILTER_JOB) ==="
V1_XF=$(submit "$SCRIPTS/v1_baseline/train_transformer.sbatch"); echo "  transformer: $V1_XF"
V1_RNN=$(submit "$SCRIPTS/v1_baseline/train_rnn.sbatch");       echo "  rnn:         $V1_RNN"
V1_AR=$(submit "$SCRIPTS/v1_baseline/train_rnn_ar.sbatch");     echo "  rnn_ar:      $V1_AR"

echo "=== v1_residual/smart_motion (parallel) ==="
V1R_XF=$(submit "$SCRIPTS/v1_residual/train_transformer.sbatch"); echo "  transformer: $V1R_XF"
V1R_RNN=$(submit "$SCRIPTS/v1_residual/train_rnn.sbatch");       echo "  rnn:         $V1R_RNN"
V1R_AR=$(submit "$SCRIPTS/v1_residual/train_rnn_ar.sbatch");     echo "  rnn_ar:      $V1R_AR"

echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R'
