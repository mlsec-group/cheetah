#!/bin/bash
#SBATCH --job-name=cheetah-metrics-train
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=4
#SBATCH --output=data/logs/cheetah-metrics-train/array_%A-%a.out
#SBATCH --array=0-9

set -e
pwd; cat /etc/hostname; date

# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/apptainer_cheetah.sh

if [ ${slurm:=0} -eq 1 ]; then
  mkdir -p "${log_dir}/cheetah-metrics-train"
  _cheetah_run auto.sif python -m metrics.train \
      --data-dir "${data_dir}/${dataset}/mcpp-microsoft__unixcoder-base-nine" \
      --job-id "${SLURM_ARRAY_TASK_ID}"
else
  for job_id in 0 1 2 3 4 5 6 7 8 9; do
    echo "Running job ${job_id}"
    _cheetah_run auto.sif python -m metrics.train \
        --data-dir "${data_dir}/${dataset}/mcpp-microsoft__unixcoder-base-nine" \
        --job-id "${job_id}"
  done
fi

date
