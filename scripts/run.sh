#!/bin/bash
# Runs the experiments, each stage after the stages it depends on.
#
#   ./scripts/run.sh                 everything
#   ./scripts/run.sh <experiment>…   only the named experiments:
#                                    dataset vulnerability_evaluation metrics causal_analysis
#                                    prompting (never by default: paid OpenAI API,
#                                    needs OPENAI_API_KEY and internet access)
#
# Run ./scripts/build.sh first. An experiment that is not selected is assumed
# to be finished already.
#
#   slurm=0 ./scripts/run.sh         run in this shell instead of submitting
#   DRY_RUN=1 ./scripts/run.sh       print what would be submitted
#   CHEETAH_ARRAY=0-0 ...            restrict every array job to this range

set -euo pipefail
# shellcheck source=/dev/null
. scripts/config.sh
# shellcheck source=/dev/null
. scripts/lib/submit.sh

selected=" ${*:-dataset vulnerability_evaluation metrics causal_analysis} "
want() { [[ "${selected}" == *" $1 "* ]]; }

ve=scripts/vulnerability_evaluation
me=scripts/metrics
ca=scripts/causal_analysis
llms=(unixcoder codebert pdbert codegen starcoder)

dataset_job="" finetune_jobs="" llm_done="" calculate_job=""

if want dataset; then
  dataset_job=$(submit scripts/dataset/run.sh)
fi

if want vulnerability_evaluation; then
  for llm in "${llms[@]}"; do
    finetune=$(submit "${ve}/llm_finetune/${llm}.sh" --after "${dataset_job}")
    evaluate=$(submit "${ve}/llm_eval/${llm}.sh" --after "${finetune}")
    aggregate=$(submit "${ve}/llm_aggregate/${llm}.sh" --after "${evaluate}")
    finetune_jobs=$(join_ids "${finetune_jobs}" "${finetune}")
    llm_done=$(join_ids "${llm_done}" "${aggregate}")
  done

  graphs=$(submit "${ve}/graphs/preprocess.sh")
  gnn=$(submit "${ve}/gnn.sh" --after "${graphs}")
  submit "${ve}/gnn_aggregate.sh" --after "${gnn}" > /dev/null

  submit "${ve}/sast.sh" --after "${dataset_job}" > /dev/null
fi

if want metrics; then
  calculate_job=$(submit "${me}/calculate.sh" --after "${dataset_job}")
  isolation=$(submit "${me}/isolation.sh" --after "${calculate_job}")
  leaveoneout=$(submit "${me}/leaveoneout.sh" --after "${calculate_job}")
fi

if want prompting; then
  batches=$(submit scripts/prompting/batches.sh --after "${dataset_job}")
  responses=$(submit scripts/prompting/submit.sh --after "${batches}")
  submit scripts/prompting/classify.sh --after "${responses}" > /dev/null
fi

causal_done=""
if want causal_analysis; then
  preprocess=$(submit "${ca}/preprocess.sh" --after "$(join_ids "${finetune_jobs}" "${calculate_job}")")
  per_model=$(submit "${ca}/access_per_model.sh" --after "${preprocess}")
  access=$(submit "${ca}/access.sh" --after "${per_model}")
  cross=$(submit "${ca}/cross.sh" --after "${preprocess}")
  cross_aggregate=$(submit "${ca}/cross_aggregate.sh" --after "${cross}")
  correlation=$(submit "${ca}/correlation.sh" --after "$(join_ids "${preprocess}" "${llm_done}")")
  dependence=$(submit "${ca}/dependence.sh" --after "${preprocess}")
  causal_done=$(join_ids "${access}" "${cross_aggregate}" "${correlation}" "${dependence}")
fi

if want metrics; then
  # metrics.train writes into hf-model/<model>/<uid>/, where the LLM evaluation
  # and several causal-analysis steps expect checkpoints only, so it runs once
  # those have ended.
  train=$(submit "${me}/train.sh" --after "${calculate_job}" --after-any "$(join_ids "${llm_done}" "${causal_done}")")
  submit "${me}/aggregate.sh" --after "$(join_ids "${train}" "${isolation}" "${leaveoneout}")" > /dev/null
fi

if [ "${slurm}" -eq 1 ]; then
  echo "[+] Submitted. Logs: ${log_dir}/<job>/. Results are printed in the logs of the aggregate jobs."
fi
