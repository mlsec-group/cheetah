#!/bin/bash

set -eu

sif_file="/home/$USER/gnn-baselines.sif"
volume_dir="/home/$USER/cheetah/data"
volumes="primevul"

bind_locations=(
    "/home/$USER/gnn-baselines/docker/gnn-baselines:/gnn-baselines"
)
for volume in ${volumes}; do
    bind_locations+=( "${volume_dir}/${volume}:/gnn-baselines/data/${volume}" )
done

bind_args=()
for location in ${bind_locations[@]}; do
    bind_args+=("--bind ${location}")
done

container_cmd="cd /gnn-baselines && source .venv/bin/activate && ./eval_devign.sh /gnn-baselines/data/primevul /gnn-baselines/data/primevul/models"

pushd docker > /dev/null
pushd gnn-baselines > /dev/null
    apptainer run ${bind_args[@]} --nv --writable-tmpfs $sif_file "$@" bash -c "${container_cmd}"
popd > /dev/null
popd > /dev/null

