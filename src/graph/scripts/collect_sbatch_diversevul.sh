#!/bin/bash

set -eu

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tag>"
    exit 1
fi

tag=$1
DATA_ROOT=$HOME/cheetah/data/diversevul-projects
SLURM_LOGS_PATH=$HOME/gnn-baselines/slurm-logs/$tag-diversevul
OUT_PATH=$HOME/cheetah/out/$tag-diversevul

rm -rf $OUT_PATH
mkdir -p $OUT_PATH/slurm-logs
mkdir -p $OUT_PATH/models

cp -r $DATA_ROOT/models/$tag-* $OUT_PATH/models
cp $DATA_ROOT/models/w2v.kv $OUT_PATH/models
cp -r $SLURM_LOGS_PATH $OUT_PATH

tar -czf $OUT_PATH.tar.gz -C $(dirname $OUT_PATH) $(basename $OUT_PATH)
rm -rf $OUT_PATH
