<<<<<<< HEAD
source scripts/config.sh

if [ ${slurm:=0} -eq 1 ]; then
  apptainer run auto.sif python3 src/metrics/train.py data/mcpp-full ${SLURM_ARRAY_TASK_ID}
else
  for job_id in {0..45}; do
    apptainer run auto.sif python3 src/metrics/train.py data/mcpp-full ${job_id}
=======
#!/bin/sh
#SBATCH --job-name=underground-metrics
#SBATCH --partition=cpu-2h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --cpus-per-task=4
#SBATCH --output=data/output/underground-metrics/array_%A-%a.out
#SBATCH --array=0-9

pwd; cat /etc/hostname; date

source scripts/config.sh

if [ ${slurm:=0} -eq 1 ]; then
  apptainer run container.sif python3 src/metrics/train.py --data-dir data/mcpp-full --job-id ${SLURM_ARRAY_TASK_ID}
else
  for job_id in {0..9} do
    apptainer run container.sif python3 src/metrics/train.py --data-dir data/mcpp-full --job-id ${job_id}
>>>>>>> a6ab224 (Update automl training)
  done
fi
