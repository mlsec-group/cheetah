#!/usr/bin/env bash

set -eu

srun --partition=cpu-9m bash -c 'cd ~/gnn-baselines/apptainer && ./build.sh'