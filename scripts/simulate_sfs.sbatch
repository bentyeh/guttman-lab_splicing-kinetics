#!/bin/bash
#
# Simulate spliced fraction
# 
#SBATCH --cpus-per-task=1
#SBATCH --array=0-80
#SBATCH --job-name=simulate_sfs
#SBATCH --mem-per-cpu=8000
#SBATCH --time=02:00:00
#SBATCH --output=/central/groups/guttman/btyeh/splicing-kinetics/logs/simulate_sfs/slurm-%A_%a.out
#SBATCH --error=/central/groups/guttman/btyeh/splicing-kinetics/logs/simulate_sfs/slurm-%A_%a.err

dir_project='/central/groups/guttman/btyeh/splicing-kinetics'
source ~/.bashrc
conda activate py3
python "${dir_project}/scripts/simulate_sfs.py" $SLURM_ARRAY_TASK_ID