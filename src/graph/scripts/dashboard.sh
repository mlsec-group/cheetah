#!/bin/bash

pushd docker > /dev/null
docker compose up gnn-baselines-dashboard
popd
