# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reproduction code for the paper "LLM-based Vulnerability Discovery through the Lens of Code Metrics". It compares fine-tuned LLMs, GNNs, SAST tools and a classifier on 23 syntactic code metrics on PrimeVul, then tests how strongly LLM predictions depend on those metrics. There is no package install, no test suite and no linter: Bash scripts run `python -m <module>` inside Apptainer images, as Slurm jobs.

## Hard rule: experiment code is frozen

The Python and Joern code under `src/` is authoritative. Do not change anything that could affect an experimental outcome, even an apparent bug: record it in `docs/reproduction-report.md` instead. Allowed without asking: scripting, automation, performance, refactoring that provably keeps behaviour. A patch to experiment code needs explicit approval from an author and an entry in `scripts/dev/approved_patches.txt`.

Run `./scripts/dev/check_python_frozen.sh` before every commit; it fails if `src/**/*.py`, `*.sc` or `src/graph/cpgtools/joern_scripts/` differ from the baseline commit without approval.

## Running things

All scripts assume **CWD = repository root** (`scripts/config.sh` and `src/paths.py` both derive paths from it).

```bash
git submodule update --init                # sast/ (SAST tool images, mlsec-group/sast)
./scripts/build.sh                         # images into .sif/, PrimeVul, PDBERT, SAST (needs internet)
./scripts/run.sh                           # submit every stage with Slurm job dependencies
./scripts/run.sh metrics causal_analysis   # only these experiments (others assumed finished)
DRY_RUN=1 ./scripts/run.sh                 # print the sbatch commands
slurm=0 CHEETAH_ARRAY=0-0 ./scripts/run.sh dataset   # run locally, array jobs restricted to task 0
SLURM_ARRAY_TASK_ID=3 bash scripts/metrics/calculate.sh   # one task of one stage, no Slurm
```

A single module by hand:

```bash
. scripts/config.sh; . scripts/lib/run_in_sif.sh
_cheetah_run --nv --env-file "${hf_env_file}" "${llm_sif}" \
    python -m llm.finetune_bertlike "${data_dir}/${dataset}" microsoft/codebert-base 0
```

Shell checks: `uvx --from shellcheck-py shellcheck -x -S warning -e SC2154,SC2034 $(git ls-files 'scripts/*.sh' 'scripts/**/*.sh')`.

## Script architecture

- `scripts/config.sh`: paths, one variable per image (`llm_sif`, `metrics_sif`, `cpgtools_sif`, `gnn_sif`), `base_models` (the five LLMs; array jobs over models index into it), `model_slug`, `slurm` (default 1).
- `scripts/lib/run_in_sif.sh`: `_cheetah_run [apptainer options] <image.sif> <command...>` sets `PYTHONPATH=${PWD}/src`; the repo is never installed into an image.
- `scripts/lib/submit.sh`: `submit <script> [--after ids] [--after-any ids]` submits with `sbatch` (`afterok` / `afterany`) or, with `slurm=0`, runs the script once per array task. It creates the Slurm log directories, which must exist before a job starts.
- `scripts/run.sh` is the only orchestrator and encodes the stage graph (diagram in `docs/reproducibility.md`).
- A stage script has its own `#SBATCH` header and handles exactly one array task (`SLURM_ARRAY_TASK_ID`, default 0). The array index is the run id (`0-9`), the model index (`metrics/calculate.sh`), the metric (`isolation`, `leaveoneout`) or model × seed (`gnn.sh`). Do not add loops over the array range or Slurm/non-Slurm branches to stage scripts.

## Images

| Image | Modules |
|---|---|
| `llm` | `dataset.dataset`, `llm.*`, `metrics.calculate`, `causal_analysis.*`, `sast.*`, `prompting.*` (CUDA base, Python 3.11) |
| `metrics` | `metrics.train`, `isolation`, `leaveoneout`, `aggregate` (Python 3.9, auto-sklearn) |
| `cpgtools` | Joern: `graph.cpgtools.preprocess.{parse,export}` |
| `gnn` | `graph.datasets.primevul`, `graph.train`, `graph.evaluate`, `graph.collect` (PyTorch/PyG) |

Each `.def` copies its pip requirements from `src/<pkg>/*-requirements.txt`; `build.sh` skips images that already exist, so delete the `.sif` to rebuild.

## Pipeline and data layout

Stages hand over through directories under `data/<dataset>/`, not through Python imports. The packages in `src/` only share `src/paths.py`.

```
raw/primevul_{train,valid,test}.jsonl
  -> hf/                                   dataset.dataset (splits train/val/test; columns texts, labels)
  -> mcpp-<slug>/                          metrics.calculate (23 metrics, paper names S1..T3)
       autosklearn-model-{isolation,leaveoneout}/
  -> hf-model/<slug>/<uid>/checkpoint-*/   llm.finetune_*, then results.json/predictions.json from llm.eval_*
       checkpoint-*/embedded/              llm.add_embeddings
       autosklearn-model-10min/            metrics.train
  -> <slug>/<uid>/checkpoint-*/            llm.add_predictions (also under data/primevul-lookaround)
  -> gnn/...                               graph pipeline, runs in gnn/runs/seed-<n>/<model>
  -> prompting/{raw-batches,batches,cache,out}/   prompting.* (GPT-3.5/GPT-4o via the OpenAI Batch API)
```

- The prompting experiment (`scripts/prompting/`) is opt-in (`./scripts/run.sh prompting`): it is paid and needs `OPENAI_API_KEY` from the environment. Never write the key into a file. `src/prompting/` was ported from the authors' archive and is frozen at the commit that added it.

- `model_slug` turns `org/name` into `org__name`; the local PDBERT checkpoint `data/pdbert` is `data__pdbert`. Several analysis modules hardcode these slugs.
- `metrics.train` writes into the LLM run directories, where other modules expect only `checkpoint-*`; `run.sh` therefore runs it last, and `metrics/aggregate.sh` aggregates it through a symlink view.
- `llm.cross_information` and `causal_analysis.cross` resolve paths against the working directory; their scripts use `--pwd data/primevul`. `causal_analysis.dependence` hardcodes `./data/primevul-lookaround`.
- `data/primevul-lookaround` (four versions per function) is used by the dependence test and is not created by this repository.
- Metric ids and the MI / mRMR orderings: `src/metrics/config.py`; mapping to mcpp names: top of `src/metrics/calculate.py`. The reported score is VD-S at 0.5 % FPR.

## Known state

`docs/reproduction-report.md` is the list of deviations from the paper setup, apparent defects left untouched, script choices awaiting author confirmation, and proposed patches for modules that cannot run against what upstream modules write (`llm.cross_information`, `causal_analysis.dependence`, `causal_analysis.correlation`). Read it before touching a causal-analysis stage, and add to it instead of fixing experiment code.
