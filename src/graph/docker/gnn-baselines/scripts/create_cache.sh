#!/usr/bin/env bash

set -eu

if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name=$1
EXTRA_ARGS="experiment.dataset=\"${dataset_name}\" fixed_params.max_epochs=0"

for model in devign linear_mcpp unixcoder; do
    gnn-baselines train start experiment.model_name=$model $EXTRA_ARGS
done
