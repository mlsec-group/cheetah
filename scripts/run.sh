#!/bin/bash
set -e
# Prepare the dataset (optional):
# ./scripts/dataset/run.sh
#
# Run vulnerability evaluation (optional):
# ./scripts/vulnerability_evaluation/run.sh
#
# Metric analysis:
./scripts/metric_analysis/run.sh

# Causal / root-cause analysis pipeline:
./scripts/causal_analysis/run.sh
