source scripts/config.sh

if [ ${slurm:=0} -eq 1 ]; then
  apptainer run auto.sif python3 src/metrics/train.py data/mcpp-full ${SLURM_ARRAY_TASK_ID}
else
  for job_id in {0..45}; do
    apptainer run auto.sif python3 src/metrics/train.py data/mcpp-full ${job_id}
  done
fi
