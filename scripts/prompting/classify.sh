#!/bin/bash
#SBATCH --job-name=cheetah-prompting-classify
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=data/logs/cheetah-prompting-classify/%j.out
#SBATCH --error=data/logs/cheetah-prompting-classify/%j.err

# Prints F1, AUPRC, MCC, BAcc and VD-S (mean and 90 % CI over the
# repetitions) of each prompted model.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

for model in "${gpt_models[@]}"; do
  echo "[+] ${model}"
  _cheetah_run "${llm_sif}" python -m prompting.classify "${data_dir}/${dataset}" "${model}" --size s
done
