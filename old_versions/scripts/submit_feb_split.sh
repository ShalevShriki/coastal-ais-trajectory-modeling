#!/bin/bash
# Submit the February pipeline as dependent SLURM jobs.
#   - East Coast is already finalized (503,351 windows) -> skipped.
#   - Mexican Coast: finalize-only (28 day parquets on disk, streamed window build).
#   - West Coast: full download + finalize (streamed window build).
#   - Combine + filter: runs only after Mexican AND West succeed.
set -euo pipefail

SCRIPTS=/home/projects/crml-prj10844/deep_learning/project/proj/project/scripts

MEX=$(sbatch --parsable "$SCRIPTS/feb_mexican_finalize.sbatch")
echo "Submitted Mexican finalize: $MEX"

WEST=$(sbatch --parsable "$SCRIPTS/feb_west_download.sbatch")
echo "Submitted West download+finalize: $WEST"

COMBINE=$(sbatch --parsable --dependency="afterok:${MEX}:${WEST}" "$SCRIPTS/feb_combine_filter.sbatch")
echo "Submitted Combine+filter (after $MEX,$WEST): $COMBINE"

echo ""
echo "Queued jobs:"
squeue -u "$USER"
