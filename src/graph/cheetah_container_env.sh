# shellcheck shell=bash
# Default SIF paths for graph pipelines. Source from repo root: . src/graph/cheetah_container_env.sh
# Override GNN_PREPROCESS_SIF / GNN_TRAINING_SIF before sourcing if images live elsewhere.

: "${GNN_PREPROCESS_SIF:=${PWD}/gnn-preprocessing.sif}"
: "${GNN_TRAINING_SIF:=${PWD}/gnn-training.sif}"
