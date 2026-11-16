#!/bin/bash
#SBATCH --job-name=cheetah-aggregate-metrics
#SBATCH --partition=cpu-9m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=data/logs/cheetah-aggregate-metrics/%j.out
#SBATCH --error=data/logs/cheetah-aggregate-metrics/%j.err

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/run_in_sif.sh

mcpp_dir="${data_dir}/${dataset}/mcpp-microsoft__unixcoder-base-nine"

echo "== Metrics classifier, all 23 metrics =="
# metrics.train writes to hf-model/<model>/<uid>/autosklearn-model-10min.
# llm.aggregate_results expects <dir>/<uid>/results.json, so give it a view
# of symlinks with that layout.
for base_model in "${base_models[@]}"; do
  slug="$(model_slug "${base_model}")"
  view="${data_dir}/${dataset}/autosklearn-view/${slug}"
  rm -rf "${view}"
  mkdir -p "${view}"
  for run_dir in "${data_dir}/${dataset}/hf-model/${slug}"/*/autosklearn-model-10min; do
    [ -d "${run_dir}" ] || continue
    ln -s "${run_dir}" "${view}/$(basename "$(dirname "${run_dir}")")"
  done
  echo "-- ${slug}"
  _cheetah_run "${llm_sif}" python -m llm.aggregate_results "${view}"
done

echo "== Leave-one-out =="
_cheetah_run "${metrics_sif}" python -m metrics.aggregate --result-dir "${mcpp_dir}/autosklearn-model-leaveoneout"

echo "== Isolation =="
_cheetah_run "${metrics_sif}" python -m metrics.aggregate --result-dir "${mcpp_dir}/autosklearn-model-isolation"
