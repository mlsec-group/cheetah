#!/bin/bash

data_dir=$(pwd)/data
log_dir=$(pwd)/logs
hf_cache_dir=${data_dir}/.hf-cache
slurm=0
apptainer_cachedir=$(pwd)/tmp
dataset=primevul

mkdir -p ${hf_cache_dir} ${log_dir}