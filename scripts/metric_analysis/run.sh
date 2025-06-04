#!/bin/bash

source scripts/config.sh

set -e

if [ ${slurm:=0} -eq 1 ]; then
  sbatch --wait train.sh
  sbatch --wait isolation.sh
  sbatch --wait leaveoneout.sh
else
  ./train.sh
  ./isolation.sh
  ./leaveoneout.sh
fi

python3 eval.py

