#!/bin/bash

source scripts/config.sh

apptainer run container.sif python3 src/metrics/calculate.py --data-dir ${data_dir} --dataset-name ${dataset} --no-truncate --base_model microsoft/unixcoder-base-nine
