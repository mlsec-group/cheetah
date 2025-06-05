#!/bin/bash

for i in $(seq 1 10);
do
    apptainer run container.sif python3 src/llm/cross_information.py data/mcpp microsoft/unixcoder-base-nine $i
    apptainer run container.sif python3 src/llm/cross_information.py data/mcpp microsoft/codebert-base $i
    apptainer run container.sif python3 src/llm/cross_information.py data/mcpp data/pdbert $i
    apptainer run container.sif python3 src/llm/cross_information.py data/mcpp Salesforce/codegen25-7b-multi_P $i
    apptainer run container.sif python3 src/llm/cross_information.py data/mcpp bigcode/starcoder2-7b $i
done

# collect results
apptainer run container.sif python3 src/causal_analysis/cross.py