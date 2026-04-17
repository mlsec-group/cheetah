#!/bin/bash
set -euo pipefail
# shellcheck source=/dev/null
. scripts/apptainer_cheetah.sh

_cheetah_run container.sif python -m causal_analysis.correlation
