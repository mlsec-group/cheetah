#!/bin/bash

. scripts/config.sh

set -e


if [ ${slurm:=0} -eq 1 ]; then
 sbatch --wait scripts/metric_analysis/calculate.sh
 sbatch --wait scripts/metric_analysis/train.sh
 sbatch --wait scripts/metric_analysis/isolation.sh
 sbatch --wait scripts/metric_analysis/leaveoneout.sh
else
 ./scripts/metric_analysis/calculate.sh
 ./scripts/metric_analysis/train.sh
 ./scripts/metric_analysis/isolation.sh
 ./scripts/metric_analysis/leaveoneout.sh
fi

python3 src/metrics/eval.py --data-dir ${data_dir}
