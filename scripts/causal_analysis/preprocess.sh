#!/bin/bash
#SBATCH --job-name=cheetah-causal-analysis-preprocess
#SBATCH --partition=gpu-5h
#SBATCH --constraint=80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --output=data/logs/causal-analysis-preprocess/%j.out
#SBATCH --error=data/logs/causal-analysis-preprocess/%j.err
#SBATCH --exclude=head022,head076

set -euo pipefail
# CWD must be repository root (interactive or sbatch).
# shellcheck source=/dev/null
. scripts/apptainer_cheetah.sh

PV="${PWD}/data/primevul"
PVL="${PWD}/data/primevul-lookaround"
PDB="${PWD}/data/pdbert"

HF_ENV="${PWD}/huggingface.env"
for _mcpp_dataset_root in "${PV}" "${PVL}"; do
  _mcpp_data_dir=$(dirname -- "${_mcpp_dataset_root}")
  _mcpp_dataset_name=$(basename -- "${_mcpp_dataset_root}")
  _cheetah_run --env-file "${HF_ENV}" container.sif python -m metrics.calculate \
    --data-dir "${_mcpp_data_dir}" \
    --dataset-name "${_mcpp_dataset_name}" \
    --no-truncate \
    --base-model microsoft/unixcoder-base-nine
done

for i in $(seq 1 10); do
  _cheetah_run container.sif python -m llm.add_predictions "${PV}" microsoft/unixcoder-base-nine "${i}"
done

for i in $(seq 1 10); do
  _cheetah_run container.sif python -m llm.add_predictions "${PVL}" microsoft/unixcoder-base-nine "${i}"
done

for i in $(seq 1 10); do
  _cheetah_run container.sif python -m llm.add_embeddings "${PV}" microsoft/unixcoder-base-nine "${i}"
done
