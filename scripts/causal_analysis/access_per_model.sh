#!/bin/bash
#SBATCH --job-name=cheetah-causal-access-per-model
#SBATCH --output=data/logs/causal-access-per-model/%A-%a.out
#SBATCH --error=data/logs/causal-access-per-model/%A-%a.err
#SBATCH --mem=40G
#SBATCH --partition=cpu-2d
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-9

# One array task per run id: regress each code metric on the embeddings.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

uid="${SLURM_ARRAY_TASK_ID:-0}"
for base_model in "${base_models[@]}"; do
  _cheetah_run --env-file "${hf_env_file}" "${llm_sif}" python -m causal_analysis.access_per_model \
      "${data_dir}/${dataset}" "${base_model}" "${uid}"
done
