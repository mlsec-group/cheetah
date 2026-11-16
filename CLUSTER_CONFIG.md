# Cluster configuration

What to adapt to run the experiments on a Slurm cluster other than the one the authors used. Everything cluster-specific is in the `#SBATCH` headers of the stage scripts listed below and in a few settings files. The Python code has no cluster-specific settings.

## Checklist

1. **Partitions.** Replace the partition names (`cpu-2h`, `cpu-5h`, `cpu-2d`, `cpu-9m`, `gpu-2h`, `gpu-2d`) with your own. No script sets `--time`, so each job gets the default time limit of its partition. If your partitions have different defaults, add `#SBATCH --time=…`.
2. **GPU selection.** GPU jobs request `--gpus-per-task=1` and `--constraint=80gb` (a GPU with 80 GB of memory, a feature name on the original cluster). Replace the constraint with your cluster's feature or GRES syntax. Remove `--exclude=head022,head076`; it names two nodes of the original cluster.
3. **Memory and CPUs.** Keep at least the values below. `metrics/calculate.sh` and `graphs/preprocess.sh` pass the number of allocated CPUs to the code, so fewer CPUs only make them slower.
4. **Internet access.** See [Internet access](#internet-access). If your compute nodes are offline, do those steps on a login node.
5. **Paths.** See [Settings outside the Slurm headers](#settings-outside-the-slurm-headers).
6. **Check.** `DRY_RUN=1 ./scripts/run.sh` prints every `sbatch` call with its dependencies without submitting anything.

## Stage scripts

All paths are relative to the repository root, which must also be the working directory when submitting. Every script also sets `--nodes=1`, `--ntasks=1`, `--job-name` and `--output`/`--error` under `data/logs/<job>/`; `scripts/lib/submit.sh` creates those log directories before submitting.

Abbreviations: Mem = `--mem`, or `--mem-per-cpu` marked "/CPU"; GPU = `--gpus-per-task`; C80 = `--constraint=80gb`; Excl = `--exclude=head022,head076`.

### Dataset and build

| Script | Partition | CPUs | Mem | GPU | Array | Notes |
|---|---|---|---|---|---|---|
| [`scripts/build.sh`](scripts/build.sh) | cpu-5h | 8 | 16G | | | Not submitted by `run.sh`; usually run directly on a node with internet access. Needs `apptainer build --fakeroot`. |
| [`scripts/dataset/run.sh`](scripts/dataset/run.sh) | cpu-9m | 8 | 16G | | | |

### Vulnerability evaluation

| Script | Partition | CPUs | Mem | GPU | Array | Notes |
|---|---|---|---|---|---|---|
| [`llm_finetune/unixcoder.sh`](scripts/vulnerability_evaluation/llm_finetune/unixcoder.sh) | gpu-2d | 4 | 40G | 1 | 0-9 | C80, Excl |
| [`llm_finetune/codebert.sh`](scripts/vulnerability_evaluation/llm_finetune/codebert.sh) | gpu-2d | 4 | 40G | 1 | 0-9 | C80, Excl |
| [`llm_finetune/pdbert.sh`](scripts/vulnerability_evaluation/llm_finetune/pdbert.sh) | gpu-2d | 4 | 40G | 1 | 0-9 | C80, Excl |
| [`llm_finetune/codegen.sh`](scripts/vulnerability_evaluation/llm_finetune/codegen.sh) | gpu-2d | 4 | 40G | 1 | 0-9 | C80, Excl. 7B model, per-device batch 8: needs the 80 GB GPU. |
| [`llm_finetune/starcoder.sh`](scripts/vulnerability_evaluation/llm_finetune/starcoder.sh) | gpu-2d | 4 | 80G | 1 | 0-9 | C80, Excl. 7B model, per-device batch 8: needs the 80 GB GPU. |
| [`llm_eval/unixcoder.sh`](scripts/vulnerability_evaluation/llm_eval/unixcoder.sh) | gpu-2h | 4 | 40G | 1 | 0-9 | C80, Excl |
| [`llm_eval/codebert.sh`](scripts/vulnerability_evaluation/llm_eval/codebert.sh) | gpu-2h | 4 | 40G | 1 | 0-9 | C80, Excl |
| [`llm_eval/pdbert.sh`](scripts/vulnerability_evaluation/llm_eval/pdbert.sh) | gpu-2h | 4 | 40G | 1 | 0-9 | C80, Excl |
| [`llm_eval/codegen.sh`](scripts/vulnerability_evaluation/llm_eval/codegen.sh) | gpu-2h | 4 | 80G | 1 | 0-9 | C80, Excl |
| [`llm_eval/starcoder.sh`](scripts/vulnerability_evaluation/llm_eval/starcoder.sh) | gpu-2h | 4 | 200G | 1 | 0-9 | C80, Excl |
| [`llm_aggregate/unixcoder.sh`](scripts/vulnerability_evaluation/llm_aggregate/unixcoder.sh) | cpu-9m | 1 | 16G | | | |
| [`llm_aggregate/codebert.sh`](scripts/vulnerability_evaluation/llm_aggregate/codebert.sh) | cpu-9m | 1 | 16G | | | |
| [`llm_aggregate/pdbert.sh`](scripts/vulnerability_evaluation/llm_aggregate/pdbert.sh) | cpu-9m | 1 | 16G | | | |
| [`llm_aggregate/codegen.sh`](scripts/vulnerability_evaluation/llm_aggregate/codegen.sh) | cpu-9m | 1 | 16G | | | |
| [`llm_aggregate/starcoder.sh`](scripts/vulnerability_evaluation/llm_aggregate/starcoder.sh) | cpu-9m | 1 | 16G | | | |
| [`graphs/preprocess.sh`](scripts/vulnerability_evaluation/graphs/preprocess.sh) | cpu-2d | 64 | 8gb/CPU | | | Joern parse and export; 512 GB in total |
| [`gnn.sh`](scripts/vulnerability_evaluation/gnn.sh) | gpu-2d | 4 | 40G | 1 | 0-19 | Devign and ReVeal × 10 seeds; no 80 GB constraint |
| [`gnn_aggregate.sh`](scripts/vulnerability_evaluation/gnn_aggregate.sh) | cpu-9m | 1 | 16G | | | |
| [`sast.sh`](scripts/vulnerability_evaluation/sast.sh) | cpu-2h | 4 | 40gb/CPU | | | Runs the SAST images from `./sast` |

### Code metrics

| Script | Partition | CPUs | Mem | GPU | Array | Notes |
|---|---|---|---|---|---|---|
| [`metrics/calculate.sh`](scripts/metrics/calculate.sh) | cpu-2d | 64 | 200G | | 0-4 | One task per LLM tokenizer. Metric extraction has a 30 s timeout per function, so heavily oversubscribed nodes change results. |
| [`metrics/isolation.sh`](scripts/metrics/isolation.sh) | cpu-2h | 4 | 4gb/CPU | | 0-229 | 10-minute auto-sklearn budget per task (wall clock). Add a `%N` throttle to the array range if your cluster limits array sizes. |
| [`metrics/leaveoneout.sh`](scripts/metrics/leaveoneout.sh) | cpu-2h | 4 | 4gb/CPU | | 0-229 | As isolation |
| [`metrics/train.sh`](scripts/metrics/train.sh) | cpu-5h | 4 | 4gb/CPU | | 0-9 | As isolation |
| [`metrics/aggregate.sh`](scripts/metrics/aggregate.sh) | cpu-9m | 1 | 16G | | | |

The auto-sklearn budget is wall-clock time, so slower or shared CPUs give the classifier fewer model evaluations. Don't give these jobs fewer CPUs than listed.

### Causal analysis

| Script | Partition | CPUs | Mem | GPU | Array | Notes |
|---|---|---|---|---|---|---|
| [`causal_analysis/preprocess.sh`](scripts/causal_analysis/preprocess.sh) | gpu-2d | 4 | 160G | 1 | 0-9 | C80, Excl. Predictions and embeddings of all five LLMs |
| [`causal_analysis/access_per_model.sh`](scripts/causal_analysis/access_per_model.sh) | cpu-2d | 4 | 40G | | 0-9 | |
| [`causal_analysis/access.sh`](scripts/causal_analysis/access.sh) | cpu-9m | 4 | 40G | | | |
| [`causal_analysis/cross.sh`](scripts/causal_analysis/cross.sh) | cpu-2d | 8 | 160G | | 0-9 | |
| [`causal_analysis/cross_aggregate.sh`](scripts/causal_analysis/cross_aggregate.sh) | cpu-9m | 1 | 16G | | | |
| [`causal_analysis/correlation.sh`](scripts/causal_analysis/correlation.sh) | cpu-9m | 4 | 40G | | | |
| [`causal_analysis/dependence.sh`](scripts/causal_analysis/dependence.sh) | cpu-2h | 4 | 40G | | | |

### Prompting (opt-in)

| Script | Partition | CPUs | Mem | GPU | Array | Notes |
|---|---|---|---|---|---|---|
| [`prompting/batches.sh`](scripts/prompting/batches.sh) | cpu-2h | 1 | 16G | | | |
| [`prompting/submit.sh`](scripts/prompting/submit.sh) | cpu-2d | 1 | 4G | | 0-1 | Needs internet access and `OPENAI_API_KEY`; waits up to the Batch API's 24 h completion window |
| [`prompting/classify.sh`](scripts/prompting/classify.sh) | cpu-2h | 1 | 8G | | | |

[`scripts/slurm/wrapper_example.sbatch`](scripts/slurm/wrapper_example.sbatch) is an old example and is not used by `run.sh`.

## Settings outside the Slurm headers

| File | Setting | Adapt when |
|---|---|---|
| [`scripts/config.sh`](scripts/config.sh) | `data_dir` (`<repo>/data`), `sif_dir` (`<repo>/.sif`, overridable from the environment), `apptainer_cachedir` (`~/.cache/apptainer/tmp`, build cache and temporary directory) | Your home or repository file system is small or slow. `data_dir` must stay `<repo>/data`, because `src/paths.py` assumes that layout; use a symlink to move it. |
| [`huggingface.env`](huggingface.env) | `HF_HOME` (model and dataset cache; passed into the containers) and `HF_HUB_DISABLE_XET` | Always: it points into the original cluster's file system. Put it on a file system all nodes can reach. |
| [`scripts/build.sh`](scripts/build.sh) | `apptainer build --fakeroot`; with `slurm=1` it binds `/dev/shm` to `/tmp` during builds | Your cluster has no fakeroot (build elsewhere and copy the `.sif` files to `.sif/`) or a small `/dev/shm`. |
| [`scripts/lib/submit.sh`](scripts/lib/submit.sh) | `sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:…/afterany:…` | Your Slurm version or site policy rejects one of these options. |
| [`scripts/lib/run_in_sif.sh`](scripts/lib/run_in_sif.sh) | `apptainer exec`; GPU stages add `--nv` | You use Singularity instead of Apptainer, or a GPU runtime other than NVIDIA. |

## Internet access

| Step | Needs |
|---|---|
| `git submodule update --init` | GitHub (`sast/`) |
| `scripts/build.sh` | Docker Hub and PyPI (image builds), GitHub (Joern), the PrimeVul download, Zenodo (PDBERT) |
| `llm_finetune/*.sh`, `llm_eval/*.sh`, `metrics/calculate.sh`, `causal_analysis/preprocess.sh`, `graphs/preprocess.sh`, `gnn.sh` | Hugging Face Hub, unless the models are already in `HF_HOME`. On offline compute nodes, download the four hub models (`scripts/config.sh: base_models`; PDBERT is local) and `microsoft/unixcoder-base-nine` for the GNN node features into `HF_HOME` on a login node first. |
| `prompting/submit.sh` | api.openai.com |

## Running without Slurm

`slurm=0 ./scripts/run.sh …` runs every stage in the current shell, one array task after the other, and ignores all headers. `CHEETAH_ARRAY=0-0` restricts every array to its first task. That is enough for a smoke test; the full experiments need GPUs and the parallelism of a cluster.
