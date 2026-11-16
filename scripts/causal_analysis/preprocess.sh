#!/bin/bash
#SBATCH --job-name=cheetah-causal-analysis-preprocess
#SBATCH --partition=gpu-2d
#SBATCH --constraint=80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --output=data/logs/causal-analysis-preprocess/%A-%a.out
#SBATCH --error=data/logs/causal-analysis-preprocess/%A-%a.err
#SBATCH --exclude=head022,head076
#SBATCH --array=0-9

# One array task per run id: predictions on both datasets and embeddings of
# every fine-tuned model.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

export PYTHONUNBUFFERED=1

PV="${data_dir}/${dataset}"
PVL="${data_dir}/primevul-lookaround"
uid="${SLURM_ARRAY_TASK_ID:-0}"

for model in "${base_models[@]}"; do
  echo "[+] add_predictions ${dataset} uid=${uid} model=${model}"
  _cheetah_run --nv --env-file "${hf_env_file}" "${llm_sif}" \
      python -m llm.add_predictions "${PV}" "${model}" "${uid}"

  echo "[+] add_predictions primevul-lookaround uid=${uid} model=${model} (models from ${dataset})"
  _cheetah_run --nv --env-file "${hf_env_file}" "${llm_sif}" \
      python -m llm.add_predictions "${PVL}" "${model}" "${uid}" --models-dir "${PV}"

  echo "[+] add_embeddings ${dataset} uid=${uid} model=${model}"
  _cheetah_run --nv --env-file "${hf_env_file}" "${llm_sif}" \
      python -m llm.add_embeddings "${PV}" "${model}" "${uid}"
done
