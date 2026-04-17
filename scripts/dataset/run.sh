#!/bin/bash
set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/apptainer_cheetah.sh

_cheetah_run container.sif python -m dataset.dataset --data "${data_dir}" --dataset-name "${dataset}"
