#!/bin/bash

cmd="python experiments/01_hparam_tuning.py devign experiments/results/devign-study.json"

pushd docker > /dev/null
docker compose run --rm gnn-baselines ${cmd}
popd
