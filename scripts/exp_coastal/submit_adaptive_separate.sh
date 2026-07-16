#!/bin/bash
# Submit separate-encoder adaptive gate experiments (softmax then hard, parallel)
set -euo pipefail
SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_coastal

echo "=== diff-encoder adaptive: softmax + hard (parallel) ==="
J1=$(sbatch --parsable "$SCRIPTS/train_adaptive_separate_softmax.sbatch")
J2=$(sbatch --parsable "$SCRIPTS/train_adaptive_separate_hard.sbatch")
echo "Softmax gate: $J1"
echo "Hard gate:    $J2"
echo ""
squeue -u "$USER" -o '%.8i %.14j %.2t %.10M %.20R' 2>/dev/null || true
