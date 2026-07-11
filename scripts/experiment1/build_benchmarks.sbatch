#!/bin/bash
#SBATCH --job-name=e1_bench
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp1_benchmark-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp1_benchmark-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

echo "Job started: $(date)"
PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
cd "$SUBROOT" || exit 1

UNKNOWN="data/results/USA Combined/unknown"

for RECIPE in v1 v2; do
  BASE="$UNKNOWN/$RECIPE/experiment1"
  FLAT=$(find "$BASE" -name lstm_metrics.json 2>/dev/null | head -1)
  AR=$(find "$BASE" -name lstm_ar_metrics.json 2>/dev/null | head -1)
  TRANS=$(find "$BASE" -name transformer_metrics.json 2>/dev/null | head -1)
  OUT="$BASE/training_benchmark.json"
  echo "=== $RECIPE/experiment1 ==="
  echo " flat=$FLAT"
  echo " ar=$AR"
  echo " xf=$TRANS"
  mkdir -p "$BASE"
  $PYTHON scripts/build_training_benchmark.py \
    --label "experiment1_${RECIPE}" \
    --description "Feb fixed lon; ${RECIPE}/experiment1; 400k sample" \
    --output "$OUT" \
    ${FLAT:+--flat-lstm "$FLAT"} \
    ${AR:+--lstm-ar "$AR"} \
    ${TRANS:+--transformer "$TRANS"}
  echo "Wrote $OUT"
done

echo "Job ended: $(date)"
