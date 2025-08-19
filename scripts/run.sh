#!/bin/bash

set -e 
# Prepare the dataset
./scripts/dataset/run.sh

# Run the vulnerability discovery performance evaluation
./scripts/vulnerability_evaluation/run.sh

## Run metric analysis experiment
./scripts/metric_analysis/run.sh

## Perform the root cause analysis
./scripts/root_cause/run.sh
