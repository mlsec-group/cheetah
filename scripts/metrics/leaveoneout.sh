#!/bin/bash
#SBATCH --job-name=cheetah-metrics-leaveoneout
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=4
#SBATCH --output=data/logs/cheetah-metrics-leaveoneout/array_%A-%a.out
#SBATCH --array=0-229

# One array task per (metric, run): the module trains on metric
# SLURM_ARRAY_TASK_ID % 23 and writes run SLURM_ARRAY_TASK_ID, so 0-229 gives
# 10 runs per metric.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

_cheetah_run "${metrics_sif}" python -m metrics.leaveoneout \
    --data-dir "${data_dir}/${dataset}/mcpp-microsoft__unixcoder-base-nine" \
    --job-id "${SLURM_ARRAY_TASK_ID:-0}"
