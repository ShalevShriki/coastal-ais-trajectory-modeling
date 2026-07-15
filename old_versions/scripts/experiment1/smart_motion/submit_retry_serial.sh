#!/bin/bash
# Retry failed smart_motion jobs serially (one GPU at a time).
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/experiment1/smart_motion

echo "=== Serial retry: v1 AR -> v1r Transformer -> v1r RNN ==="
J1=$(sbatch --parsable "$SCRIPTS/v1_baseline/train_rnn_ar.sbatch")
echo "1/3 v1 RNN_AR: $J1"

J2=$(sbatch --parsable --dependency=afterok:$J1 "$SCRIPTS/v1_residual/train_transformer.sbatch")
echo "2/3 v1_residual Transformer (after $J1): $J2"

J3=$(sbatch --parsable --dependency=afterok:$J2 "$SCRIPTS/v1_residual/train_rnn.sbatch")
echo "3/3 v1_residual RNN (after $J2): $J3"

echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R'
