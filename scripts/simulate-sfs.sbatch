#!/bin/bash

# Simulate spliced fraction

dir_project='/central/groups/guttman/btyeh/splicing-kinetics'

source ~/.bashrc
conda activate py3
python "${dir_project}/scripts/simulate-sfs.py" $SLURM_ARRAY_TASK_ID