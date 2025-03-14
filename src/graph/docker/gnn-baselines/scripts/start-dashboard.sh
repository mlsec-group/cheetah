#!/usr/bin/env bash

set -eu

optuna-dashboard --host 0.0.0.0 postgresql://dbuser:password@db:5432/gnn-baselines
