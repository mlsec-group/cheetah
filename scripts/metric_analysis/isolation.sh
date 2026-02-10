#!/bin/sh
#SBATCH --job-name=cheetah-metrics-sf
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=4
#SBATCH --output=data/output/cheetah-metrics-sf/array_%A-%a.out
#SBATCH --array=0-45

pwd; cat /etc/hostname; date

source scripts/config.sh

if [ ${slurm:=0} -eq 1 ]; then
  apptainer run auto.sif python3 src/metrics/isolation.py --data-dir data/mcpp-subset --job-id ${SLURM_ARRAY_TASK_ID}
else
  for job_id in {0..45}; do
    apptainer run auto.sif python3 src/metrics/isolation.py --data-dir data/mcpp-subset --job-id ${job_id}
  done
fi

date
