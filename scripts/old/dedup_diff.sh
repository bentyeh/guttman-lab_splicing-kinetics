#!/bin/bash
#
#SBATCH --cpus-per-task=4
#SBATCH --array=0,10,15,20,25,30,45,60,75,90,120,240
#SBATCH --job-name=dedup_diff_no_flags_or_tags
#SBATCH --mem-per-cpu=12000
#SBATCH --time=4:00:00
#SBATCH --output=/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup/slurm-diff_no_flags_or_tags-%A_%a.out
#SBATCH --error=/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup/slurm-diff_no_flags_or_tags-%A_%a.err

# set -e
# set -o pipefail

source ~/.bashrc
module load samtools/1.15

dir_logs='/central/groups/guttman/btyeh/splicing-kinetics/logs/dedup'
dir_data_dedup='/central/groups/guttman/btyeh/splicing-kinetics/data/dedup'
dir_scratch='/central/scratch/btyeh'

# file_picard="${dir_data_dedup}/10.MarkDuplicates.bam"
# file_samtools="${dir_data_dedup}/10.markdup.bam"

timepoint="$SLURM_ARRAY_TASK_ID"
file_picard="${dir_data_dedup}/${timepoint}.MarkDuplicates.bam"
file_samtools="${dir_data_dedup}/${timepoint}.markdup.bam"

# create diff
comm -23 --nocheck-order \
    <(samtools view -@ 10 "$file_picard" | 
        cut -f 1,2,12- |
        # remove Picard-specific tags
        sed -E -e 's/\sPG:Z:MarkDuplicates//' -e 's/DT:Z:/dt:Z:/' |
        # sort remaining tags
        awk -v FS='\t' -v OFS='\t' '{
            printf "%s\t%s", $1, $2;
            $1="";
            $2="";
            split($0, arr);
            asort(arr);
            for (i=1; i<=length(arr); i++) {
                if (arr[i] != "") {
                    printf "\t%s", arr[i]}
            };
            printf "\n"}' |
        sort -T "$dir_scratch") \
    <(samtools view -@ 10 "$file_samtools" | 
        cut -f 1,2,12- |
        # remove samtools-specific tags
        sed -E -e 's/\sMC:Z:\S+//' -e 's/\sMQ:i:\S+//' -e 's/\sms:i:\S+//' |
        # sort remaining tags
        awk -v FS='\t' -v OFS='\t' '{
            printf "%s\t%s", $1, $2;
            $1="";
            $2="";
            split($0, arr);
            asort(arr);
            for (i=1; i<=length(arr); i++) {
                if (arr[i] != "") {
                    printf "\t%s", arr[i]}
            };
            printf "\n"}' |
        sort -T "$dir_scratch") \
    > "${dir_data_dedup}/${timepoint}.picard_diff.sam"

# comm -13 --nocheck-order \
#     <(samtools view -@ 10 "$file_picard" | 
#         # remove Picard-specific tags
#         sed -E -e 's/\sPG:Z:MarkDuplicates//' |
#         # sort remaining tags
#         {
#             while read i; do
#                 fields=$(echo "$i" | cut -f 1-11)
#                 tags=$(echo -n "$i" | cut -f 12- | tr '\t' '\n' | sort | tr '\n' '\t' | sed -E 's/\s+$//g')
#                 echo -e "${fields}\t${tags}"
#             done
#         } |
#         sort) \
#     <(samtools view "$file_samtools" | 
#         # remove samtools-specific tags
#         sed -E -e 's/\sMC:Z:\S+//' -e 's/\sMQ:i:\S+//' -e 's/\sms:i:\S+//' |
#         # sort remaining tags; create key without flags field
#         {
#             while read i; do
#                 fields=$(echo "$i" | cut -f 1-11)
#                 tags=$(echo -n "$i" | cut -f 12- | tr '\t' '\n' | sort | tr '\n' '\t' | sed -E 's/\s+$//g')
#                 echo -e "${fields}\t${tags}"
#             done
#         } |
#         sort) \
#     > "${dir_data_dedup}/10.samtools_diff.sam"

        # {
        #     while read i; do
        #         fields=$(echo "$i" | cut -f 1-2)
        #         tags=$(echo -n "$i" | cut -f 3- | tr '\t' '\n' | sort | tr '\n' '\t' | sed -E 's/\s+$//g')
        #         echo -e "${fields}\t${tags}"
        #     done
        # } |
