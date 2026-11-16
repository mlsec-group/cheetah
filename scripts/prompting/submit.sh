#!/bin/bash
#SBATCH --job-name=cheetah-prompting-submit
#SBATCH --partition=cpu-2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
#SBATCH --output=data/logs/cheetah-prompting-submit/%A_%a.out
#SBATCH --error=data/logs/cheetah-prompting-submit/%A_%a.err

# Submits the request file of model gpt_models[task] 10 times to the OpenAI
# Batch API (paid; needs internet access) and waits for the responses
# (prompting/out/). Uploads and batch ids are cached in prompting/cache/, so a
# rerun resumes the same batches instead of paying again.
#
# The API key is read from OPENAI_API_KEY in the environment; never write it
# into a file in this repository.

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in the environment}"
model="${gpt_models[${SLURM_ARRAY_TASK_ID:-0}]}"

_cheetah_run "${llm_sif}" python -m prompting.submit "${data_dir}/${dataset}" "${model}" --size s --repetitions 10
