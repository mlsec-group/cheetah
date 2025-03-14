#!/bin/bash

pushd docker > /dev/null
docker compose run --rm gnn-baselines ./run_all.sh
popd
