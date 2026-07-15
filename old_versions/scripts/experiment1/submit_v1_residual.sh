#!/bin/bash
# v1 recipe + --residual-naive only (no motion-balanced sampling).
# Outputs: data/results/USA Combined/unknown/v1_residual/experiment1/<Model>/
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/experiment1/v1_residual

echo "=== v1_residual/experiment1 (parallel, separate from v1/experiment1) ==="
XF=$(sbatch --parsable "$SCRIPTS/train_transformer.sbatch"); echo "v1_residual transformer: $XF"
RNN=$(sbatch --parsable "$SCRIPTS/train_rnn.sbatch");       echo "v1_residual rnn:         $RNN"
AR=$(sbatch --parsable "$SCRIPTS/train_rnn_ar.sbatch");     echo "v1_residual ar:          $AR"

echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R'
