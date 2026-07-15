#!/bin/bash
#SBATCH --job-name=ec_filter
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_coastal_filter-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/exp_coastal_filter-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj
set -euo pipefail
source /home/projects/crml-prj10844/deep_learning/project/proj/project/scripts/exp_coastal/_env.sh
cd "$SUBROOT"
echo "Inland filter | started $(date)"
$PYTHON -u scripts/filter_inland_windows.py \
  --input data/processed/combined_filtered_smart/train.parquet \
  --output "$DATA" \
  --open-water-km 10 \
  --inland-fraction 0.5 \
  --subsample-every 6
# Prebuild land grid for training penalty
$PYTHON -u -c "
from pathlib import Path
from proj.project.models.land_mask_utils import build_or_load_land_grid
p = Path('data/processed/land_grid_us.npz')
g = build_or_load_land_grid(p)
print('land grid', g['land'].shape, 'cached at', p)
"
echo "ended $(date)"
