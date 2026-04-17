#!/bin/bash
set -euo pipefail
# shellcheck source=/dev/null
. scripts/apptainer_cheetah.sh

for i in $(seq 1 10); do
  _cheetah_run container.sif python -m llm.cross_information "${PWD}/data/mcpp" microsoft/unixcoder-base-nine "${i}"
  _cheetah_run container.sif python -m llm.cross_information "${PWD}/data/mcpp" microsoft/codebert-base "${i}"
  _cheetah_run container.sif python -m llm.cross_information "${PWD}/data/mcpp" "${PWD}/data/pdbert" "${i}"
  _cheetah_run container.sif python -m llm.cross_information "${PWD}/data/mcpp" Salesforce/codegen25-7b-multi_P "${i}"
  _cheetah_run container.sif python -m llm.cross_information "${PWD}/data/mcpp" bigcode/starcoder2-7b "${i}"
done

_cheetah_run container.sif python -m causal_analysis.cross
