#!/bin/bash
#SBATCH --job-name=feb_rebuild_win
#SBATCH --output=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_rebuild_windows-%j.out
#SBATCH --error=/home/projects/crml-prj10844/deep_learning/project/proj/project/LOG/feb_rebuild_windows-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --partition=part-ugproj
#SBATCH --qos=qos-ugproj

# Rebuild Feb model_ready_windows from coastal_segments only (no re-download).
# Expected wall: ~1.5–3h sequential (East ~30m, West ~35m, Mex ~50–70m).

echo "Job started: $(date)"
echo "Host: $(hostname)"

PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
PROJECT=/home/projects/crml-prj10844/deep_learning/project
SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"

cd "$SUBROOT" || { echo "ERROR: subroot not found"; exit 1; }

echo "=== Rebuild Feb windows from coastal_segments (streaming, lat/lon fix) ==="
$PYTHON -u scripts/rebuild_feb_windows_from_segments.py

echo "Job ended: $(date)"
