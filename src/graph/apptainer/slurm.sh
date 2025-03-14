#!/bin/bash -eu
#SBATCH --job-name=cheetah-reveal
#SBATCH --partition=gpu-2h
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=slurm-logs/cheetah-reveal-%j.out

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

model=$1

/home/$USER/gnn-baselines/apptainer/run_${model}.sh ${model}