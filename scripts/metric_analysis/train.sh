#!/bin/sh
#SBATCH --job-name=cheetah-metrics
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=4
#SBATCH --output=data/output/cheetah-metrics/array_%A-%a.out
#SBATCH --array=0-9

set -e
pwd; cat /etc/hostname; date

. scripts/config.sh

if [ ${slurm:=0} -eq 1 ]; then
  apptainer run container.sif python3 src/metrics/train.py --data-dir ${data_dir}/mcpp-full --job-id ${SLURM_ARRAY_TASK_ID}
else
  for job_id in 0 1 2 3 4 5 6 7 8 9; do
    echo "Running job ${job_id}"
    apptainer run auto.sif python3 src/metrics/train.py --data-dir data/primevul/mcpp-microsoft__unixcoder-base-nine --job-id ${job_id}
  done
fi
