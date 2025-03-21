#!/bin/bash

# Compute relevant outputs of trained models
./scripts/causal_analysis/preprocess.sh

# Compute information access
./scripts/causal_analysis/access.sh

# Compute information correlation
./scripts/causal_analysis/correlation.sh

# Compute information dependence
./scripts/causal_analysis/dependence.sh

# Compute cross information
./scripts/causal_analysis/cross.sh