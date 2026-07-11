#!/bin/bash
# Resubmit failed/stale v3 jobs with 192G memory and chain benchmark.
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts

# Cancel stale benchmark and optionally running jobs to pick up 192G mem.
scancel 21993 2>/dev/null || true

# Resubmit all three models (192G in sbatch files).
RNN=$(sbatch --parsable "$SCRIPTS/train_rnn_usa_v3.sbatch")
echo "Submitted Flat RNN v3 (192G): $RNN"

AR=$(sbatch --parsable "$SCRIPTS/train_rnn_ar_usa_v3.sbatch")
echo "Submitted AR RNN v3 (192G): $AR"

TRANS=$(sbatch --parsable "$SCRIPTS/train_transformer_usa_v3.sbatch")
echo "Submitted Transformer v3 (192G): $TRANS"

BENCH=$(sbatch --parsable --dependency="afterok:${RNN}:${AR}:${TRANS}" "$SCRIPTS/build_benchmark_v3.sbatch")
echo "Submitted benchmark v3 (after $RNN,$AR,$TRANS): $BENCH"

echo ""
squeue -u "$USER"
