#!/bin/bash
#SBATCH --job-name=cheetah-prompting-batches
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=data/logs/cheetah-prompting-batches/%j.out
#SBATCH --error=data/logs/cheetah-prompting-batches/%j.err

# Draws the stratified test subsets (prompting/raw-batches/) and writes one
# Batch API request file per model (prompting/batches/). No API access.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

_cheetah_run "${llm_sif}" python -m prompting.raw_batches "${data_dir}/${dataset}"
for model in "${gpt_models[@]}"; do
  _cheetah_run "${llm_sif}" python -m prompting.batches "${data_dir}/${dataset}" "${model}" --size s
done
