#!/bin/bash
# Full recovery pipeline after lat=lon window bug:
#   1) rebuild Feb windows from coastal_segments
#   2) combine + history-only filter
#   3) lean serial training (Transformer → RNN → AR → benchmark)
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts

# Cancel stale train/bench jobs that would use corrupt data
scancel -u "$USER" -n ais_transformer_v3 2>/dev/null || true
scancel -u "$USER" -n ais_rnn_v3 2>/dev/null || true
scancel -u "$USER" -n ais_rnn_ar_v3 2>/dev/null || true
scancel -u "$USER" -n bench_v3 2>/dev/null || true
scancel -u "$USER" -n feb_rebuild_win 2>/dev/null || true
scancel -u "$USER" -n feb_combine 2>/dev/null || true
sleep 1

REBUILD=$(sbatch --parsable "$SCRIPTS/rebuild_feb_windows.sbatch")
echo "1) Rebuild Feb windows: $REBUILD"

COMBINE=$(sbatch --parsable --dependency=afterok:${REBUILD} "$SCRIPTS/feb_combine_filter.sbatch")
echo "2) Combine + filter (after $REBUILD): $COMBINE"

TRANS=$(sbatch --parsable --dependency=afterok:${COMBINE} "$SCRIPTS/train_transformer_usa_v3.sbatch")
echo "3) Transformer (after $COMBINE): $TRANS"

RNN=$(sbatch --parsable --dependency=afterok:${TRANS} "$SCRIPTS/train_rnn_usa_v3.sbatch")
echo "4) Flat RNN (after $TRANS): $RNN"

AR=$(sbatch --parsable --dependency=afterok:${RNN} "$SCRIPTS/train_rnn_ar_usa_v3.sbatch")
echo "5) AR RNN (after $RNN): $AR"

BENCH=$(sbatch --parsable --dependency=afterok:${TRANS}:${RNN}:${AR} "$SCRIPTS/build_benchmark_v3.sbatch")
echo "6) Benchmark (after train trio): $BENCH"

echo ""
squeue -u "$USER"
