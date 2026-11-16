#!/bin/bash
#SBATCH --job-name=cheetah-metrics-train
#SBATCH --partition=cpu-5h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=4
#SBATCH --output=data/logs/cheetah-metrics-train/%A-%a.out
#SBATCH --error=data/logs/cheetah-metrics-train/%A-%a.err
#SBATCH --array=0-9

# One array task per run id: metrics classifier (auto-sklearn) for each
# model's tokenisation.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

uid="${SLURM_ARRAY_TASK_ID:-0}"
for base_model in "${base_models[@]}"; do
  echo "[+] metrics.train ${base_model} uid=${uid}"
  _cheetah_run "${metrics_sif}" python -m metrics.train \
      --data-dir "${data_dir}" \
      --dataset-name "${dataset}" \
      --base-model "${base_model}" \
      --uid "${uid}"
done
