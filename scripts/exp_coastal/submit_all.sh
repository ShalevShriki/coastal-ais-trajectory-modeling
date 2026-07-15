#!/bin/bash
# Coastal-only: filter inland → full model suite (does NOT overwrite exp_final)
set -euo pipefail
SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_coastal

echo "=== exp_coastal full suite ==="
J0=$(sbatch --parsable "$SCRIPTS/filter_inland.sbatch"); echo "Filter inland: $J0"
J1=$(sbatch --parsable --dependency=afterok:$J0 "$SCRIPTS/train_AR_9h.sbatch");  echo "AR 9h:         $J1"
J2=$(sbatch --parsable --dependency=afterok:$J1 "$SCRIPTS/train_AR_12h.sbatch"); echo "AR 12h:        $J2"
J3=$(sbatch --parsable --dependency=afterok:$J2 "$SCRIPTS/train_AR_18h.sbatch"); echo "AR 18h:        $J3"
J4=$(sbatch --parsable --dependency=afterok:$J3 "$SCRIPTS/train_AR_24h.sbatch"); echo "AR 24h:        $J4"
J5=$(sbatch --parsable --dependency=afterok:$J4 "$SCRIPTS/train_flat_lstm.sbatch"); echo "Flat LSTM:     $J5"
J6=$(sbatch --parsable --dependency=afterok:$J5 "$SCRIPTS/train_transformer.sbatch"); echo "Transformer:   $J6"
J7=$(sbatch --parsable --dependency=afterok:$J6 "$SCRIPTS/train_adaptive.sbatch"); echo "Adaptive AR:   $J7"
J8=$(sbatch --parsable --dependency=afterok:$J7 "$SCRIPTS/train_sliding_3h.sbatch"); echo "Sliding 3h:    $J8"
echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R' 2>/dev/null || true
