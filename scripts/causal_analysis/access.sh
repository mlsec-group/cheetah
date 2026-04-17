#!/bin/bash
#SBATCH --job-name=cheetah-causal-access-per-model
#SBATCH --output=data/logs/causal-access-per-model/%j.out
#SBATCH --error=data/logs/causal-access-per-model/%j.err
#SBATCH --mem=40G
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
# shellcheck source=/dev/null
. scripts/apptainer_cheetah.sh

for i in $(seq 1 10); do
  _cheetah_run container.sif python -m causal_analysis.access_per_model "${PWD}/data/mcpp" microsoft/unixcoder-base-nine "${i}"
done

_cheetah_run container.sif python -m causal_analysis.access
