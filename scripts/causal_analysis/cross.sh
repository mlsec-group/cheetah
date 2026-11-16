#!/bin/bash
#SBATCH --job-name=cheetah-causal-cross
#SBATCH --output=data/logs/causal-cross/%A-%a.out
#SBATCH --error=data/logs/causal-cross/%A-%a.err
#SBATCH --mem=160G
#SBATCH --partition=cpu-2d
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-9

# One array task per run id: classifier on embeddings with and without the
# code metrics.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

uid="${SLURM_ARRAY_TASK_ID:-0}"
# llm.cross_information resolves hf-model/ and mcpp-<model> against the
# working directory, so run it inside the dataset directory.
for base_model in "${base_models[@]}"; do
  _cheetah_run --pwd "${data_dir}/${dataset}" "${llm_sif}" python -m llm.cross_information \
      mcpp "${base_model}" "${uid}"
done
