#!/bin/bash
#SBATCH --job-name=bench_v3
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/build_benchmark_v3-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/build_benchmark_v3-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# Merge v3 metrics JSONs into training_benchmark_improved_v3.json.
# Finds the newest lstm_metrics / lstm_ar_metrics / transformer_metrics under USA Combined.

echo "Job started: $(date)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"

cd "$SUBROOT" || exit 1

RESULTS_ROOT="data/results/USA Combined"
FLAT=$(find "$RESULTS_ROOT" -name lstm_metrics.json -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
AR=$(find "$RESULTS_ROOT" -name lstm_ar_metrics.json -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
TRANS=$(find "$RESULTS_ROOT" -name transformer_metrics.json -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "Flat LSTM metrics: $FLAT"
echo "AR LSTM metrics:   $AR"
echo "Transformer:       $TRANS"

OUT="data/results/USA Combined/unknown/training_benchmark_improved_v3.json"

$PYTHON scripts/build_training_benchmark.py \
    --label improved_v3_feb \
    --description "February combined_filtered; split-first; motion-balanced; residual+kinematic; nFDE/nADE" \
    --output "$OUT" \
    ${FLAT:+--flat-lstm "$FLAT"} \
    ${AR:+--lstm-ar "$AR"} \
    ${TRANS:+--transformer "$TRANS"}

echo "Wrote $OUT"
echo "Job ended: $(date)"
