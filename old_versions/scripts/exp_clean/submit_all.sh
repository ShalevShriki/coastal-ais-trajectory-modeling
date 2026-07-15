#!/bin/bash
# Submit exp_clean suite serially (no residual experiments).
set -euo pipefail
SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_clean

echo "Cancelling legacy ar_exp jobs (if any)..."
scancel -u "$USER" -n ar_exp_flat -n ar_exp_A -n ar_exp_B -n ar_exp_C -n ar_exp_D -n ar_exp_E 2>/dev/null || true

echo "=== exp_clean serial: B1 -> B2 -> A0 -> A1 (residual runs removed) ==="
J1=$(sbatch --parsable "$SCRIPTS/train_B1_flat.sbatch"); echo "B1 flat:        $J1"
J2=$(sbatch --parsable --dependency=afterok:$J1 "$SCRIPTS/train_B2_transformer.sbatch"); echo "B2 transformer: $J2"
J3=$(sbatch --parsable --dependency=afterok:$J2 "$SCRIPTS/train_A0_ar_no_tc.sbatch"); echo "A0 AR no TC:    $J3"
J4=$(sbatch --parsable --dependency=afterok:$J3 "$SCRIPTS/train_A1_ar_tc.sbatch"); echo "A1 AR + TC:     $J4"

echo ""
echo "Context experiments: bash scripts/exp_context/submit_all.sh $J4"
echo "Compare exp_clean: python scripts/compare_exp_clean.py"
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R' 2>/dev/null || true
