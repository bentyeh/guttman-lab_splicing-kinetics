#!/bin/bash

dir_project='/central/groups/guttman/btyeh/splicing-kinetics'

sbatch \
    --cpus-per-task=1 \
    --array=0-80 \
    --job-name='simulate-bcs' \
    --mem-per-cpu=8000 \
    --time="02:00:00" \
    --output="${dir_project}/logs/simulate-sfs/slurm-%A_%a.out" \
    --error="${dir_project}/logs/simulate-sfs/slurm-%A_%a.err" \
    "${dir_project}/scripts/simulate-bcs.sbatch"