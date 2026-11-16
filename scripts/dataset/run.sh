#!/bin/bash
#SBATCH --job-name=dataset-cheetah
#SBATCH --partition=cpu-9m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=data/logs/dataset-cheetah/%j.out
#SBATCH --error=data/logs/dataset-cheetah/%j.err

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

_cheetah_run "${llm_sif}" python -m dataset.dataset --data "${data_dir}" --dataset-name "${dataset}"
