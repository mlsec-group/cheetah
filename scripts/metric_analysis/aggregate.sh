#!/bin/bash
#SBATCH --job-name=cheetah-aggregate-metrics
#SBATCH --partition=cpu-9m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=data/logs/cheetah-aggregate-metrics/std.out
#SBATCH --error=data/logs/cheetah-aggregate-metrics/std.err

pwd; cat /etc/hostname; date

# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/apptainer_cheetah.sh
mkdir -p "${log_dir}/cheetah-aggregate-metrics"

echo "******************************************************"
echo "**      Aggregate metrics for baseline model        **"
echo "******************************************************"

_cheetah_run container.sif python -m metrics.aggregate \
    --result-dir "${PWD}/data/primevul/mcpp-microsoft__unixcoder-base-nine/autosklearn-model"


echo "******************************************************"
echo "**      Aggregate metrics for leave-one-out model   **"
echo "******************************************************"

_cheetah_run container.sif python -m metrics.aggregate \
    --result-dir "${PWD}/data/primevul/mcpp-microsoft__unixcoder-base-nine/autosklearn-model-leaveoneout"


echo "******************************************************"
echo "**      Aggregate metrics for isolation model       **"
echo "******************************************************"

_cheetah_run container.sif python -m metrics.aggregate \
    --result-dir "${PWD}/data/primevul/mcpp-microsoft__unixcoder-base-nine/autosklearn-model-isolation"

date
