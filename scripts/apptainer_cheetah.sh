# shellcheck shell=bash
# Source from project root:  . scripts/apptainer_cheetah.sh
# Defines run_in_sif and _cheetah_run (CWD visible in container; PYTHONPATH=\${PWD}/src).

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=/dev/null
. "${SCRIPT_DIR}/lib/run_in_sif.sh"
