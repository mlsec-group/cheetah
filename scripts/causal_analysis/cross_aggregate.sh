#!/bin/bash
#SBATCH --job-name=cheetah-causal-cross-aggregate
#SBATCH --output=data/logs/causal-cross-aggregate/%j.out
#SBATCH --error=data/logs/causal-cross-aggregate/%j.err
#SBATCH --mem=16G
#SBATCH --partition=cpu-9m
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

# causal_analysis.cross reads ./hf-model, so run it inside the dataset directory.
_cheetah_run --pwd "${data_dir}/${dataset}" "${llm_sif}" python -m causal_analysis.cross
