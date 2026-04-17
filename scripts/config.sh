#!/bin/sh
# Expect CWD = repository root. Paths are relative to pwd.

_base="$(pwd)"
data_dir="${_base}/data"
log_dir="${_base}/logs"
hf_env_file="${_base}/huggingface.env"
slurm=1
apptainer_cachedir="${_base}/tmp"
dataset=primevul

. "${hf_env_file}"
mkdir -p "${HF_HOME}" "${log_dir}"
