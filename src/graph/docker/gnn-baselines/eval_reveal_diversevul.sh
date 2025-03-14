#!/usr/bin/env bash

set -u

if [ $# -ne 2 ]; then
    echo "Usage: $0 <data_root> <model_root>"
    exit 1
fi

data_root=$1
model_root=$2

if [ ! -d ${model_root} ]; then
    echo "Model root ${model_root} does not exist"
    exit 1
fi

model_id=$SLURM_ARRAY_TASK_ID
IFS='' read -r -d '' optimal_kwargs <<"EOF"
models.reveal.hidden_dim=256
models.reveal.ggnn_layers=2
models.reveal.graph_embed_size=256
models.reveal.extractor_layers=2
models.reveal.dropout=0.3
hparams.lr=1e-4
hparams.weight_decay=0
hparams.batch_size=256
hparams.accumulated_batches=50
EOF
optimal_kwargs=$(echo "$optimal_kwargs" | tr '\n' ' ')

model_dir=${model_root}/reveal-${model_id}
if [ -d ${model_dir} ]; then
    echo "[*] clearing previous model at ${model_dir}"
    rm -rf ${model_dir}
fi
mkdir -p ${model_dir}

gnn-baselines train start \
    paths.models=${model_dir} \
    ${optimal_kwargs} \
    fixed_params.max_epochs=100 \
    fixed_params.patience=10 \
    experiment.model_name=reveal \
    experiment.seed=${model_id} \
    paths.data_root=${data_root} \
    dataset=diversevul-projects
