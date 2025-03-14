#!/bin/bash -eu
#SBATCH --job-name=cheetah-devign
#SBATCH --partition=gpu-2d
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=slurm-logs/devign/%j.out
#SBATCH --exclude=head022,head076

/home/$USER/gnn-baselines/hydra/run_devign.sh
