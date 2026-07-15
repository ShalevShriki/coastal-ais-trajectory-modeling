#!/bin/bash
# Submit AR ablation experiments A–E serially (one GPU at a time).
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/ar_experiments

echo "=== AR experiment suite (serial): flat -> A -> B -> C -> D -> E ==="

J0=$(sbatch --parsable "$SCRIPTS/train_flat.sbatch")
echo "0/6 flat baseline: $J0"

J1=$(sbatch --parsable --dependency=afterok:$J0 "$SCRIPTS/train_A.sbatch")
echo "1/6 Experiment A: $J1"

J2=$(sbatch --parsable --dependency=afterok:$J1 "$SCRIPTS/train_B.sbatch")
echo "2/6 Experiment B: $J2"

J3=$(sbatch --parsable --dependency=afterok:$J2 "$SCRIPTS/train_C.sbatch")
echo "3/6 Experiment C: $J3"

J4=$(sbatch --parsable --dependency=afterok:$J3 "$SCRIPTS/train_D.sbatch")
echo "4/6 Experiment D: $J4"

J5=$(sbatch --parsable --dependency=afterok:$J4 "$SCRIPTS/train_E.sbatch")
echo "5/6 Experiment E: $J5"

echo ""
echo "Compare when done:"
echo "  python scripts/compare_ar_experiments.py"
echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R' 2>/dev/null || true
