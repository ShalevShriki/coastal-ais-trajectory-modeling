#!/bin/bash
# Submit February filter (if needed) then v3 training for all 3 models + benchmark.
#
# Pipeline:
#   1. feb_filter_only.sbatch  -> combined_filtered (history-only stationary filter)
#   2. train_*_usa_v3.sbatch (3 parallel, after filter)
#   3. build_benchmark_v3.sbatch (after all 3 models)
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts
SUBROOT=/home/projects/crml-prj10844/deep_learning/project/proj/project

# Skip filter if February combined_filtered already exists (train newer than combined).
COMBINED_TRAIN="$SUBROOT/data/processed/combined/train.parquet"
FILTERED_TRAIN="$SUBROOT/data/processed/combined_filtered/train.parquet"

NEED_FILTER=1
if [[ -f "$FILTERED_TRAIN" && -f "$COMBINED_TRAIN" ]]; then
    if [[ "$FILTERED_TRAIN" -nt "$COMBINED_TRAIN" ]]; then
        NEED_FILTER=0
        echo "combined_filtered looks up to date — skipping filter job."
    fi
fi

if [[ "$NEED_FILTER" -eq 1 ]]; then
    FILTER=$(sbatch --parsable "$SCRIPTS/feb_filter_only.sbatch")
    echo "Submitted filter: $FILTER"
    DEP="afterok:${FILTER}"
else
    DEP=""
fi

submit_train() {
    local script="$1"
    if [[ -n "$DEP" ]]; then
        sbatch --parsable --dependency="$DEP" "$script"
    else
        sbatch --parsable "$script"
    fi
}

RNN=$(submit_train "$SCRIPTS/train_rnn_usa_v3.sbatch")
echo "Submitted Flat RNN v3: $RNN"

AR=$(submit_train "$SCRIPTS/train_rnn_ar_usa_v3.sbatch")
echo "Submitted AR RNN v3: $AR"

TRANS=$(submit_train "$SCRIPTS/train_transformer_usa_v3.sbatch")
echo "Submitted Transformer v3: $TRANS"

BENCH=$(sbatch --parsable --dependency="afterok:${RNN}:${AR}:${TRANS}" "$SCRIPTS/build_benchmark_v3.sbatch")
echo "Submitted benchmark v3 (after $RNN,$AR,$TRANS): $BENCH"

echo ""
squeue -u "$USER"
