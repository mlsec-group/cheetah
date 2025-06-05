#!/bin/bash

for i in $(seq 1 10);
do
    apptainer run container.sif python3 src/causal_analysis/access_per_model.py data/mcpp microsoft/unixcoder-base-nine $i
    apptainer run container.sif python3 src/causal_analysis/access_per_model.py data/mcpp microsoft/codebert-base $i
    apptainer run container.sif python3 src/causal_analysis/access_per_model.py data/mcpp data/pdbert $i
    apptainer run container.sif python3 src/causal_analysis/access_per_model.py data/mcpp Salesforce/codegen25-7b-multi_P $i
    apptainer run container.sif python3 src/causal_analysis/access_per_model.py data/mcpp bigcode/starcoder2-7b $i
done

apptainer run container.sif python3 src/causal_analysis/access.py