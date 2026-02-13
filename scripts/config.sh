#!/bin/sh

data_dir=$(pwd)/data
log_dir=$(pwd)/logs
hf_env_file=$(pwd)/huggingface.env
slurm=0
apptainer_cachedir=$(pwd)/tmp
dataset=primevul

. ${hf_env_file}
mkdir -p ${HF_HOME} ${log_dir}
