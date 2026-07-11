#!/bin/bash
# Usage: ./submit.sh [COMBINE_JOBID]
# If COMBINE_JOBID omitted, jobs start immediately (assumes filtered data ready).
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/experiment1
DEP_PREFIX=""
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  DEP_PREFIX="--dependency=afterok:${1}"
  echo "Waiting for combine/filter job: $1"
fi

submit() {
  local path="$1"
  if [[ -n "$DEP_PREFIX" ]]; then
    sbatch --parsable $DEP_PREFIX "$path"
  else
    sbatch --parsable "$path"
  fi
}

echo "=== experiment1 v1_baseline (parallel) ==="
V1_XF=$(submit "$SCRIPTS/v1_baseline/train_transformer.sbatch"); echo "v1 transformer: $V1_XF"
V1_RNN=$(submit "$SCRIPTS/v1_baseline/train_rnn.sbatch");       echo "v1 rnn:         $V1_RNN"
V1_AR=$(submit "$SCRIPTS/v1_baseline/train_rnn_ar.sbatch");     echo "v1 ar:          $V1_AR"

echo "=== experiment1 v2_residual (parallel) ==="
V2_XF=$(submit "$SCRIPTS/v2_residual/train_transformer.sbatch"); echo "v2 transformer: $V2_XF"
V2_RNN=$(submit "$SCRIPTS/v2_residual/train_rnn.sbatch");       echo "v2 rnn:         $V2_RNN"
V2_AR=$(submit "$SCRIPTS/v2_residual/train_rnn_ar.sbatch");     echo "v2 ar:          $V2_AR"

BENCH=$(sbatch --parsable --dependency=afterok:${V1_XF}:${V1_RNN}:${V1_AR}:${V2_XF}:${V2_RNN}:${V2_AR} \
  "$SCRIPTS/build_benchmarks.sbatch")
echo "benchmark (after all 6): $BENCH"

echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R'
