#!/bin/bash

#SBATCH --job-name=cheetah-metrics-calculate
#SBATCH --output=data/logs/metrics-calculate/%j.out
#SBATCH --error=data/logs/metrics-calculate/%j.err
#SBATCH --mem=200G
#SBATCH --partition=cpu-2d
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/apptainer_cheetah.sh

if [ ${slurm:=0} -eq 1 ]; then
  mkdir -p ${log_dir}/calculate
fi
_cheetah_run --env-file "${hf_env_file}" container.sif python -m metrics.calculate \
    --data-dir "${data_dir}" \
    --dataset-name "${dataset}" \
    --no-truncate \
    --base-model microsoft/unixcoder-base-nine
