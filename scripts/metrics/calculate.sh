#!/bin/bash
#SBATCH --job-name=cheetah-metrics-calculate
#SBATCH --output=data/logs/metrics-calculate/%a-%A.out
#SBATCH --error=data/logs/metrics-calculate/%a-%A.err
#SBATCH --mem=200G
#SBATCH --partition=cpu-2d
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-4

# One array task per model: code metrics of every function, after a round trip
# through that model's tokenizer, for PrimeVul and the lookaround dataset
# (causal_analysis.dependence uses each model's lookaround metrics).

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

model="${base_models[${SLURM_ARRAY_TASK_ID:-0}]}"
num_cpus="${SLURM_CPUS_PER_TASK:-$(nproc)}"

calculate() {
  echo "[+] metrics.calculate $1 ${model}"
  _cheetah_run --env-file "${hf_env_file}" "${llm_sif}" python -m metrics.calculate \
      --data-dir "${data_dir}" \
      --dataset-name "$1" \
      --no-truncate \
      --base-model "${model}" \
      --num-cpus "${num_cpus}"
  # metrics.calculate turns extraction errors into zeros; stop if nothing was extracted.
  _cheetah_run "${llm_sif}" python -c '
import sys
from datasets import load_from_disk
ds = load_from_disk(sys.argv[1])
metrics = [c for c in ds["train"].column_names if c not in ("texts", "labels")]
if not any(any(ds[split][m]) for split in ds for m in metrics):
    sys.exit("[!] all metrics are zero in " + sys.argv[1])
' "${data_dir}/$1/mcpp-$(model_slug "${model}")"
}

calculate "${dataset}"
calculate primevul-lookaround
