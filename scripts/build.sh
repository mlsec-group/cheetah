#!/bin/bash

source scripts/config.sh

# Build containers
apptainer build container.sif container.def
apptainer build auto.sif auto.def

# Download dependencies
git clone git@github.com:MNayer/sast.git
pushd sast/cppcheck &> /dev/null
./build.sh
popd
pushd sast/rats &> /dev/null
./build.sh
popd
pushd sast/semgrep &> /dev/null
./build.sh
popd

# Download dataset
mkdir -p ${data_dir}/raw
wget 'https://drive.usercontent.google.com/download?id=12b1QkCwW0SC6l9KvxSmMe4jHF7VhjwCa&confirm=t' -O ${data_dir}/raw/primevul_train.jsonl
wget 'https://drive.usercontent.google.com/download?id=1490USYtUtb5n3i3m3n2LaSfjTCiKhPoO' -O ${data_dir}/raw/primevul_valid.jsonl
wget 'https://drive.usercontent.google.com/download?id=1ABV5cIdtyNAzKlGxjW_BsFZOp9MFd-AH' -O ${data_dir}/raw/primevul_test.jsonl
