#!/bin/bash
#SBATCH --job-name=smart_filt
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/smart_motion_filter-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/smart_motion_filter-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
cd "$SUBROOT" || exit 1

echo "Smart motion filter on combined_filtered -> combined_filtered_smart"
$PYTHON scripts/apply_training_filters.py \
  --input-dir data/processed/combined_filtered \
  --output-dir data/processed/combined_filtered_smart \
  --skip-stationary \
  --smart-motion \
  --smart-min-16h-net-km 8 \
  --smart-min-last-8h-net-km 2 \
  --smart-max-loop-ratio 0.35

echo "Done: $(date)"
