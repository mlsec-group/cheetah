#!/bin/bash

data_dir=$(pwd)/data
log_dir=$(pwd)/logs
hf_home=${data_dir}/.hf-cache
slurm=0
apptainer_cachedir=$(pwd)/tmp
dataset=primevul

mkdir -p ${hf_home} ${log_dir}

# disable xet for hf cache in NFS
HF_ENV="HF_HOME=${hf_home} HF_HUB_DISABLE_XET=1"
