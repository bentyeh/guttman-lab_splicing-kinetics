#!/bin/bash
#
# Simulate bond counts
# 
#SBATCH --cpus-per-task=4
#SBATCH --array=0-191
#SBATCH --job-name=simulate_means
#SBATCH --mem-per-cpu=10000
#SBATCH --time=24:00:00
#SBATCH --output=/central/groups/guttman/btyeh/splicing-kinetics/logs/simulate_means/slurm-%A_%a.out
#SBATCH --error=/central/groups/guttman/btyeh/splicing-kinetics/logs/simulate_means/slurm-%A_%a.err

source ~/.bashrc
conda activate py3
python "/central/groups/guttman/btyeh/splicing-kinetics/scripts/simulate_means.py" $SLURM_ARRAY_TASK_ID
