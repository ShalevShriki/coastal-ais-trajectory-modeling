#!/bin/bash
# Submit v3 lean pipeline: one model at a time, generous walltimes.
#   Transformer (4h) → RNN (3h) → AR (4h) → benchmark
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts

# Cancel any stale pending jobs from prior attempts
scancel -u "$USER" -n ais_transformer_v3 2>/dev/null || true
scancel -u "$USER" -n ais_rnn_v3 2>/dev/null || true
scancel -u "$USER" -n ais_rnn_ar_v3 2>/dev/null || true
scancel -u "$USER" -n bench_v3 2>/dev/null || true
sleep 1

TRANS=$(sbatch --parsable "$SCRIPTS/train_transformer_usa_v3.sbatch")
echo "Transformer (4h, 600k): $TRANS"

RNN=$(sbatch --parsable --dependency=afterok:${TRANS} "$SCRIPTS/train_rnn_usa_v3.sbatch")
echo "Flat RNN (3h, 800k, after $TRANS): $RNN"

AR=$(sbatch --parsable --dependency=afterok:${RNN} "$SCRIPTS/train_rnn_ar_usa_v3.sbatch")
echo "AR RNN (4h, 500k, after $RNN): $AR"

BENCH=$(sbatch --parsable --dependency=afterok:${TRANS}:${RNN}:${AR} "$SCRIPTS/build_benchmark_v3.sbatch")
echo "Benchmark (after $TRANS,$RNN,$AR): $BENCH"

echo ""
squeue -u "$USER"
