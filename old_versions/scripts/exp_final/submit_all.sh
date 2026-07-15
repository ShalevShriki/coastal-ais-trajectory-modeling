#!/bin/bash
# Final research suite per project_research.md
# Already complete (exp_clean): Flat LSTM 24h->12h (B1), Transformer 24h->12h (B2)
set -euo pipefail
SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_final

echo "=== exp_final serial: AR9 -> AR12 -> AR18 -> AR24 -> sliding3h -> adaptive ==="
echo "Baselines already done: exp_clean/B1_flat, exp_clean/B2_transformer"
echo ""

J1=$(sbatch --parsable "$SCRIPTS/train_AR_9h.sbatch");  echo "AR 9h:          $J1"
J2=$(sbatch --parsable --dependency=afterok:$J1 "$SCRIPTS/train_AR_12h.sbatch"); echo "AR 12h:         $J2"
J3=$(sbatch --parsable --dependency=afterok:$J2 "$SCRIPTS/train_AR_18h.sbatch"); echo "AR 18h:         $J3"
J4=$(sbatch --parsable --dependency=afterok:$J3 "$SCRIPTS/train_AR_24h.sbatch"); echo "AR 24h:         $J4"
J5=$(sbatch --parsable --dependency=afterok:$J4 "$SCRIPTS/train_sliding_3h.sbatch"); echo "Sliding 3h:     $J5"
J6=$(sbatch --parsable --dependency=afterok:$J5 "$SCRIPTS/train_adaptive.sbatch"); echo "Adaptive AR:    $J6"

echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R' 2>/dev/null || true
