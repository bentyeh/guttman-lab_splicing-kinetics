#!/bin/bash
#
# Deduplicate 5EU pulse-chase BAM files using Picard MarkDuplicates
# on samtools markdup-dedpulicated BAM files
# 
#SBATCH --cpus-per-task=4
#SBATCH --array=0,10,15,20,25,30,45,60,75,90,120,240
#SBATCH --job-name=dedup_samtools_picard
#SBATCH --mem-per-cpu=4000
#SBATCH --time=4:00:00
#SBATCH --output=/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup/slurm-samtools-picard-%A_%a.out
#SBATCH --error=/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup/slurm-samtools-picard-%A_%a.err

source ~/.bashrc

dir_project='/central/groups/guttman/btyeh/splicing-kinetics'
dir_data="${dir_project}/data"
dir_dedup="${dir_data}/dedup"
mkdir -p "$dir_dedup"

dir_tmp='/central/scratch/btyeh'

timepoint="$SLURM_ARRAY_TASK_ID"
directory="${timepoint}minchase"

java -jar ~/bin/picard.jar MarkDuplicates \
    -I "${dir_dedup}/${timepoint}.markdup.bam" \
    -O "${dir_dedup}/${timepoint}.markdup.MarkDuplicates.bam" \
    -M "${dir_dedup}/${timepoint}.markdup.MarkDuplicates.metrics" \
    --TMP_DIR "$dir_tmp" \
    --TAGGING_POLICY All \
    --OPTICAL_DUPLICATE_PIXEL_DISTANCE 2500
