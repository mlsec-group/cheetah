#!/bin/bash

set -e

sbatch --wait train.sh
sbatch --wait isolation.sh
sbatch --wait leaveoneout.sh

python3 eval.py
