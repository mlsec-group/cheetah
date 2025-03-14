#!/bin/sh
#SBATCH --job-name=underground-metrics
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=4
#SBATCH --output=data/output/underground-metrics/array_%A-%a.out
#SBATCH --array=0-45
pwd; hostname; date

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

apptainer run container.sif python3 src/metrics/train.py data/mcpp-full $SLURM_ARRAY_TASK_ID

date
