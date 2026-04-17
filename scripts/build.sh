#!/bin/bash

source scripts/config.sh
set -e

# Large / writable build temps (helps when host /tmp is tight or oddly permissioned).
if [ -n "${apptainer_cachedir:-}" ]; then
  mkdir -p "${apptainer_cachedir}"
  export APPTAINER_CACHEDIR="${apptainer_cachedir}"
  export APPTAINER_TMPDIR="${apptainer_cachedir}"
else
  : "${APPTAINER_TMPDIR:=${HOME}/.cache/apptainer/tmp}"
  mkdir -p "${APPTAINER_TMPDIR}"
  export APPTAINER_TMPDIR
fi

BIND_OPTS=()
if [ ${slurm:=0} -eq 1 ]; then
  BIND_OPTS+=(-B /dev/shm:/tmp)
fi

# Build containers
# apptainer build "${BIND_OPTS[@]}" dataset.sif dataset.def
# apptainer build "${BIND_OPTS[@]}" container.sif container.def
# apptainer build "${BIND_OPTS[@]}" auto.sif auto.def
apptainer build "${BIND_OPTS[@]}" gnn-preprocessing.sif gnn-preprocessing.def
# apptainer build "${BIND_OPTS[@]}" gnn-training.sif gnn-training.def
exit 0

# Download dependencies
if [ ! -d sast ]; then
  git clone https://github.com/LPirch/sast.git
fi

APPTAINER_BUILD_OPTS="${BIND_OPTS[@]}" ./sast/apptainer_build.sh semgrep
APPTAINER_BUILD_OPTS="${BIND_OPTS[@]}" ./sast/apptainer_build.sh rats
APPTAINER_BUILD_OPTS="${BIND_OPTS[@]}" ./sast/apptainer_build.sh cppcheck
APPTAINER_BUILD_OPTS="${BIND_OPTS[@]}" ./sast/apptainer_build.sh sast

# Download dataset
mkdir -p ${data_dir}/raw
wget 'https://drive.usercontent.google.com/download?id=12b1QkCwW0SC6l9KvxSmMe4jHF7VhjwCa&confirm=t' -O ${data_dir}/raw/primevul_train.jsonl
wget 'https://drive.usercontent.google.com/download?id=1490USYtUtb5n3i3m3n2LaSfjTCiKhPoO' -O ${data_dir}/raw/primevul_valid.jsonl
wget 'https://drive.usercontent.google.com/download?id=1ABV5cIdtyNAzKlGxjW_BsFZOp9MFd-AH' -O ${data_dir}/raw/primevul_test.jsonl
