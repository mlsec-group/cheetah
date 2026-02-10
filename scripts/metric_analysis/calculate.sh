#!/bin/bash

source scripts/config.sh

cmd="apptainer run container.sif python3 src/metrics/calculate.py --data-dir ${data_dir} --dataset-name ${dataset} --no-truncate --base_model microsoft/unixcoder-base-nine"

if [ ${slurm:=0} -eq 1 ]; then
    sbatch \
        --wait --partition=cpu-2h --nodes=1 --ntasks=1 --mem-per-cpu=4gb --cpus-per-task=4 \
        --output=data/output/cheetah-metrics-calculate.out \
        --error=data/output/cheetah-metrics-calculate.err \
        ${cmd}
else
    ${cmd}
fi