#!/bin/bash

#SBATCH --job-name=cheetah-calculate
#SBATCH --output=data/logs/calculate/%j.out
#SBATCH --error=data/logs/calculate/%j.err
#SBATCH --mem=200G
#SBATCH --partition=cpu-2d

. scripts/config.sh

if [ ${slurm:=0} -eq 1 ]; then
  mkdir -p ${log_dir}/calculate
fi
apptainer run --env-file ${hf_env_file} container.sif python3 src/metrics/calculate.py \
    --data-dir ${data_dir} \
    --dataset-name ${dataset} \
    --no-truncate \
    --base_model microsoft/unixcoder-base-nine