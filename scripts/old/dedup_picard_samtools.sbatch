#!/bin/bash
#
# Deduplicate 5EU pulse-chase BAM files using samtools markdup
# on Picard MarkDuplicates-dedpulicated BAM files
# 
#SBATCH --cpus-per-task=4
#SBATCH --array=0,10,15,20,25,30,45,60,75,90,120,240
#SBATCH --job-name=dedup_picard_samtools
#SBATCH --mem-per-cpu=4000
#SBATCH --time=4:00:00
#SBATCH --output=/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup/slurm-picard-samtools-%A_%a.out
#SBATCH --error=/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup/slurm-picard-samtools-%A_%a.err

source ~/.bashrc
module load samtools/1.15

dir_project='/central/groups/guttman/btyeh/splicing-kinetics'
dir_data="${dir_project}/data"
dir_dedup="${dir_data}/dedup"
mkdir -p "$dir_dedup"

dir_tmp='/central/scratch/btyeh'

timepoint="$SLURM_ARRAY_TASK_ID"
directory="${timepoint}minchase"

samtools collate -O "${dir_dedup}/${timepoint}.MarkDuplicates.bam" "${dir_tmp}/${directory}" |
    samtools fixmate -m - - |
    samtools sort -T "${dir_tmp}" -@ 4 |
    samtools markdup -T "${dir_tmp}/" --threads 4 -d 2500 -f "${timepoint}.MarkDuplicates.markdup.metrics" - "${timepoint}.MarkDuplicates.markdup.bam"
