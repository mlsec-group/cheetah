#!/bin/bash
#SBATCH --job-name=cheetah-wrapper-example
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/wrapper-example-%j.out
#SBATCH --error=logs/wrapper-example-%j.err
#
# Submit from repository root:
#   mkdir -p logs
#   sbatch scripts/slurm/wrapper_example.sbatch
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Example: replace with the phase you need (dataset, metrics, causal, …).
bash scripts/dataset/run.sh
