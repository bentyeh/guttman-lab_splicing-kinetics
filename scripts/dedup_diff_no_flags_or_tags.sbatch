#!/bin/bash
# 
# Verify that, between Picard MarkDuplicate and samtools markdup outputs,
# all fields are identical except for flags and tags.
# 
#SBATCH --cpus-per-task=4
#SBATCH --array=0,10,15,20,25,30,45,60,75,90,120,240
#SBATCH --job-name=dedup_diff_no_flags_or_tags
#SBATCH --mem-per-cpu=12000
#SBATCH --time=4:00:00
#SBATCH --output=/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup/slurm-diff_no_flags_or_tags-%A_%a.out
#SBATCH --error=/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup/slurm-diff_no_flags_or_tags-%A_%a.err

set -e
set -o pipefail

source ~/.bashrc
module load samtools/1.15

dir_logs='/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup'
dir_data_dedup='/central/groups/guttman/btyeh/splicing-kinetics/data/dedup'
dir_scratch='/central/scratch/btyeh'

timepoint="$SLURM_ARRAY_TASK_ID"

file_picard="${dir_data_dedup}/${timepoint}.MarkDuplicates.bam"
file_samtools="${dir_data_dedup}/${timepoint}.markdup.bam"
diff \
    <(samtools view -@ 10 "$file_picard" | 
        cut -f 1,3-11 |
        sort -T "$dir_scratch") \
    <(samtools view -@ 10 "$file_samtools" | 
        cut -f 1,3-11 |
        sort -T "$dir_scratch") \
> "${dir_data_dedup}/${timepoint}.no_flags_or_tags.diff"
