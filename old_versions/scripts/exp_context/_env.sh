#!/bin/bash
# Shared env for context-length experiments (project_research.md)
export PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
export PROJECT=/home/projects/crml-prj10844/deep_learning/project
export SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
export DATA=data/processed/combined_filtered_smart/train.parquet
export SAMPLE=400000
export RUN_PREFIX=exp_context
export FUTURE_H=6
export HORIZON_H=6
