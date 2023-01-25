#!/bin/bash
#
# Simulate bond counts
# 
#SBATCH --cpus-per-task=1
#SBATCH --array=0-80
#SBATCH --job-name=simulate_bcs
#SBATCH --mem-per-cpu=8000
#SBATCH --time=02:00:00
#SBATCH --output=/central/groups/guttman/btyeh/splicing-kinetics/logs/simulate_bcs/slurm-%A_%a.out
#SBATCH --error=/central/groups/guttman/btyeh/splicing-kinetics/logs/simulate_bcs/slurm-%A_%a.err

source ~/.bashrc
conda activate py3
python "/central/groups/guttman/btyeh/splicing-kinetics/scripts/simulate_bcs.py" $SLURM_ARRAY_TASK_ID