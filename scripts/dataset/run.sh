#!/bin/bash
. scripts/config.sh

apptainer run container.sif python3 src/dataset/dataset.py -d ${data_dir} --dataset-name ${dataset}
