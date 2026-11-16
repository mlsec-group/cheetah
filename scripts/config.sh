# shellcheck shell=bash
# Shared settings. Source from the repository root:  . scripts/config.sh
# All paths derive from the current directory; src/paths.py makes the same
# assumption (data lives in <repo>/data).

repo_root="$(pwd)"
if [ ! -f "${repo_root}/scripts/config.sh" ]; then
  echo "[!] Run this from the repository root." >&2
  exit 1
fi

data_dir="${repo_root}/data"
dataset=primevul
pdbert_dir="${data_dir}/pdbert"
log_dir="${data_dir}/logs"
hf_env_file="${repo_root}/huggingface.env"
apptainer_cachedir="${HOME}/.cache/apptainer/tmp"

# slurm=1: stages are submitted with sbatch. slurm=0: stages run in this shell.
slurm="${slurm:-1}"

# Container images (built by scripts/build.sh).
sif_dir="${sif_dir:-${repo_root}/.sif}"
llm_sif="${sif_dir}/llm.sif"
metrics_sif="${sif_dir}/metrics.sif"
cpgtools_sif="${sif_dir}/cpgtools.sif"
gnn_sif="${sif_dir}/gnn.sif"

# The five LLMs. Array jobs over models index into this list.
base_models=(
  microsoft/unixcoder-base-nine
  microsoft/codebert-base
  "${pdbert_dir}"
  Salesforce/codegen25-7b-multi_P
  bigcode/starcoder2-7b
)

# The prompted OpenAI models (prompting experiment). Array jobs index into it.
gpt_models=(
  gpt-3.5-turbo-0125
  gpt-4o-2024-08-06
)

# Directory name of a model under <dataset>/hf-model (mirrors src/paths.py:model_slug).
model_slug() {
  if [ "$1" = "${pdbert_dir}" ]; then
    echo "data__pdbert"
  else
    echo "${1//\//__}"
  fi
}

# shellcheck source=/dev/null
. "${hf_env_file}"
export HF_HOME
mkdir -p "${HF_HOME}" "${log_dir}" "${sif_dir}"
