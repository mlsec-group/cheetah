#!/bin/bash
#SBATCH --job-name=cheetah-causal-access
#SBATCH --output=data/logs/causal-access/%j.out
#SBATCH --error=data/logs/causal-access/%j.err
#SBATCH --mem=40G
#SBATCH --partition=cpu-9m
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

_cheetah_run "${llm_sif}" python -m causal_analysis.access "${data_dir}/${dataset}"
