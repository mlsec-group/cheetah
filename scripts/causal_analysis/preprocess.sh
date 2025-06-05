#!/bin/bash

## TODO: add mcpp preprocessing

for i in $(seq 1 10);
do
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul microsoft/unixcoder-base-nine $i
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul microsoft/codebert-base $i
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul data/pdbert $i
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul Salesforce/codegen25-7b-multi_P $i
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul bigcode/starcoder2-7b $i
done

for i in $(seq 1 10);
do
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul-lookaround microsoft/unixcoder-base-nine $i
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul-lookaround microsoft/codebert-base $i
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul-lookaround data/pdbert $i
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul-lookaround Salesforce/codegen25-7b-multi_P $i
    apptainer run container.sif python3 src/llm/add_predictions.py data/primevul-lookaround bigcode/starcoder2-7b $i
done

for i in $(seq 1 10);
do
    apptainer run container.sif python3 src/llm/add_embeddings.py data/primevul microsoft/unixcoder-base-nine $i
    apptainer run container.sif python3 src/llm/add_embeddings.py data/primevul microsoft/codebert-base $i
    apptainer run container.sif python3 src/llm/add_embeddings.py data/primevul data/pdbert $i
    apptainer run container.sif python3 src/llm/add_embeddings.py data/primevul Salesforce/codegen25-7b-multi_P $i
    apptainer run container.sif python3 src/llm/add_embeddings.py data/primevul bigcode/starcoder2-7b $i
done
