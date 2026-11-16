#!/bin/bash
#SBATCH --job-name=cheetah-causal-dependence
#SBATCH --output=data/logs/causal-dependence/%j.out
#SBATCH --error=data/logs/causal-dependence/%j.err
#SBATCH --mem=40G
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

_cheetah_run "${llm_sif}" python -m causal_analysis.dependence
