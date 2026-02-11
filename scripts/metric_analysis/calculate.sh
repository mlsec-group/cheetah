#!/bin/bash

#SBATCH --job-name=cheetah-calculate
#SBATCH --output=logs/calculate/%j.out
#SBATCH --error=logs/calculate/%j.err
#SBATCH --mem=200G
#SBATCH --partition=cpu-5h

. scripts/config.sh

if [ ${slurm:=0} -eq 1 ]; then
  mkdir -p ${log_dir}/calculate
fi

apptainer run container.sif python3 src/metrics/calculate.py \
    --data-dir ${data_dir} \
    --dataset-name ${dataset} \
    --no-truncate \
    --base_model microsoft/unixcoder-base-nine \
    --hf-cache-dir ${hf_cache_dir}
