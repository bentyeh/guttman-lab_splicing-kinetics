#!/bin/bash
#
# Fit bond counts using Gaussian processes
# 
#SBATCH --cpus-per-task=8
#SBATCH --array=0-80
#SBATCH --job-name=fitting_bcs_gp
#SBATCH --mem-per-cpu=5000
#SBATCH --time=24:00:00
#SBATCH --output=/central/groups/guttman/btyeh/splicing-kinetics/logs/fitting_bcs_gp/slurm-%A_%a.out
#SBATCH --error=/central/groups/guttman/btyeh/splicing-kinetics/logs/fitting_bcs_gp/slurm-%A_%a.err

source ~/.bashrc
conda activate py3
python "/central/groups/guttman/btyeh/splicing-kinetics/scripts/fitting_bcs_gp.py" $SLURM_ARRAY_TASK_ID
