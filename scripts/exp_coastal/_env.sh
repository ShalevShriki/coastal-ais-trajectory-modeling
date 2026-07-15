#!/bin/bash
# Coastal-only suite — inland rivers/canals removed; does not overwrite exp_final
export PYTHON=/home/projects/crml-prj10844/miniforge3/envs/compdfu_ogb_release/bin/python
export PROJECT=/home/projects/crml-prj10844/deep_learning/project
export SUBROOT=$PROJECT/proj/project
export PYTHONPATH="$PROJECT"
export DATA=data/processed/combined_filtered_smart_coastal/train.parquet
export SAMPLE=300000
export RUN_PREFIX=exp_coastal
export FUTURE_H=12
export HORIZON_H=12
export LAND_PENALTY=0.1
