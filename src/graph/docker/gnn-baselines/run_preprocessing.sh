#!/bin/bash

export SBT_HOME=/root/.sdkman/candidates/sbt/current
export JAVA_HOME=/root/.sdkman/candidates/java/current

set -eu

gnn-baselines preprocess extract dataset=primevul
gnn-baselines preprocess parse dataset=primevul
gnn-baselines preprocess export dataset=primevul
gnn-baselines preprocess build-vocab dataset=primevul

gnn-baselines train w2v dataset=primevul
