# Reproduction report

Status of the consolidation on branch `reproduction-fixes` (baseline: `reproduction` at `7b9bac1`).

Rules for this pass:

- The Python (and Joern) experiment code is authoritative and frozen at the baseline. `scripts/dev/check_python_frozen.sh` enforces this.
- Anything that could change an experimental outcome is listed here and **not** changed.
- Where Python modules contradict each other so that a stage cannot run, a minimal patch is proposed in [Proposed Python patches](#proposed-python-patches) and applied only after an author approves it. Approved patches are listed in `scripts/dev/approved_patches.txt`.

Each item is marked **[checked]** (confirmed against the file or the commit) or **[audit]** (found by a code-reading pass, not yet confirmed).

## Deviations from the paper setup

Result-changing edits made to the experiment code after the paper experiments. The final run will not match the paper's numbers exactly because of these.

| Commit | Change | Status |
|---|---|---|
| `29a1698` (2026-04-17) | GNN pipeline replaced: old Lightning code (word2vec node features, Optuna-tuned hyper-parameters, early stopping) removed; new HF-Trainer code with frozen UniXcoder node embeddings and fixed hyper-parameters. Different experiment, not a refactoring. | [audit] |
| `29a1698` | Joern export `getEdgesByLabel`: `.src.id` → `.dst.id` (`src/graph/cpgtools/joern_scripts/export.sc:143`). The old version emitted only self-loops for out-edges; every exported graph changes. | [checked] |
| `216e7fa` (2026-05-13) | `dataset/dataset.py`: val and test splits were swapped before this commit, so earlier runs selected checkpoints on PrimeVul test and reported on PrimeVul valid. | [checked] |
| `4705c79` (2026-06-04) | Effective batch size 128 → 64 for CodeGen (8×16 → 4×16) and StarCoder (4×32 → 4×16). Since approved patch A6 (2026-11-16) the layout is 8×8 as in the paper's table; still 64. | [checked] |
| `4705c79` | The 90 % train subsample in all three finetune scripts is now seeded with the run id (`shuffle(seed=uid)`); it was unseeded. | [checked] |
| `7b9bac1` (2026-11-12) | `llm/add_predictions.py` predicts with the smallest-numbered checkpoint only (before: every checkpoint). | [checked] |
| `7b9bac1` | `causal_analysis/correlation.py` loads runs `1..9` (before: the first 10 `predictions.json` found). | [checked] |
| `1babb9c` (2026-02-11) | `metrics/train.py` without `--metric-subset` trains on the 23 metrics in `BEST_MRMR`; before, it used the 28 mcpp metrics in `PRIMEVUL` (`C1–C4`, `V1–V11`, `x1–x4`, `t1–t3`, `s1–s4`, `m1`, `m2`). | [checked] |
| `29a1698` | `metrics/isolation.py`: metric index `(job_id-1) % 23` → `job_id % 23`; unseeded random padding features added. | [checked] |

The paper's hyper-parameter table (`tables/hparams.tex` in the paper sources) also disagrees with the code in two places:

| Model | Paper (`hparams.tex`) | Code |
|---|---|---|
| CodeGen 2.5, StarCoder 2 | batch 8×8, 10 epochs, LR 2e-5 | 8×8 since A6 (was 4×16), 4 epochs, LR 2e-5 (`llm/finetune_{codegen,starcoder}.py:155–164`). According to the authors, the first runs used 10 epochs, but the validation loss stopped improving after 2–3, so the paper runs used 4. The table should say 4. |
| Devign, ReVeal | batch 256×50, 100 epochs, LR 1e-4, patience 10 | 16×16, 10 epochs, LR 2e-3 (`graph/train.py:83,117–118`); the paper values probably belong to the GNN code removed in `29a1698` |

The paper calls `microsoft/codebert-base` LineVul (`causal_analysis/dependence.py:110`).

## Open author decisions

Apparent defects that affect outcomes. Left as they are.

1. *Resolved 2026-11-16 (approved patch A5):* `causal_analysis/dependence.py:111` mapped the label "StarCoder" to the CodeGen directory (`Salesforce__codegen25-7b-multi_P`), so the paper's StarCoder and CodeGen dependence rows are the same model. It now points at `bigcode__starcoder2-7b`; the StarCoder row will differ from the paper. [checked]
2. *Resolved 2026-11-13 by an author against the paper and the mcpp implementation:* metric M3 is mcpp `V5`, as in `metrics/calculate.py:43`. The `V3` in `causal_analysis/dependence.py:19` and in the metric list of `llm/cross_information.py:13` is stale; P1 and P2 remove it.
3. *Not a defect (checked 2026-11-16):* `metrics/aggregate.py` aggregates each metric directory `0..22` on its own and has a fallback for a directory `23` (all 23 metrics). No version of `isolation.py` or `leaveoneout.py` writes `23` (`metric_no = job_id % 23`), so the fallback is dead code; the all-metrics classifier comes from `metrics/train.sh`. The real problem was the array range: with `0-22` each metric directory got one run and the aggregate reported `± inf`. The scripts now run `0-229`, 10 runs per metric (before `29a1698` they ran `0-45`, two per metric). The paper's figure normalises F1 to the all-metrics classifier ("% of F1"); no code in the repository does that normalisation.
4. Unseeded randomness: `np.random.choice` over 90 % of train draws with replacement (bootstrap) in `metrics/{train,isolation,leaveoneout}.py`; random padding features; the classification head is initialised before the Trainer sets its seed (`seed=1337`, constant across runs). auto-sklearn is bounded by wall-clock (`TOTAL_TIME=600`), so results depend on CPU speed and load. [audit]
5. `metrics/calculate.py` gives all-zero metrics for a sample whose extraction exceeds 30 s; the outcome therefore depends on machine load. [audit]
6. *Resolved 2026-11-16 (approved patch A4):* run ranges disagreed: the scripts train runs `0..9`, `causal_analysis/access.py` read `0..10`, `causal_analysis/correlation.py` read `1..9`. Both now read `0..9`. [checked]
7. VD-S is computed with `roc_curve(drop_intermediate=True)` in `llm/eval_*.py` and `False` in `metrics/util.py` and `sast/classify.py`. `sast/classify.py` computes VD-S on the train split. [audit]
8. `llm/finetune_codegen.py` does not set `model.config.pad_token_id`; `finetune_starcoder.py` does. [audit]
9. GNN: no VD-S is produced; the "best" checkpoint may be the last one (custom `save_pretrained` with `load_best_model_at_end`); the train split is subsampled to 90 % twice (≈ 81 %); `task_id` does not seed weight initialisation. [audit]
10. Dependencies: several requirements are unpinned (`numpy`, `datasets`, `scikit-learn` in the LLM image; the PyG extension wheels), and only the CodeGen model revision is pinned. The package versions of the images built on 2026-11-13 are recorded in `docs/image-versions/`. [checked]
11. `llm_aggregate` passes `--last-checkpoint` for CodeBERT, CodeGen and StarCoder but not for UniXcoder and PDBERT; the comment in `dependence.py:114` says the last checkpoint was best only for the two large models. [audit]

## Proposed Python patches

Stages that cannot run because two Python modules disagree. *All approved and applied on 2026-11-16* (see [Approved Python patches](#approved-python-patches)); P2 was approved with per-model metrics instead of the UniXcoder ones described below. The complete diff is `docs/proposed-patches.diff` (`git apply docs/proposed-patches.diff`). With it applied in a scratch copy, `cross.sh`, `cross_aggregate.sh`, `access.sh`, `correlation.sh` and `dependence.sh` ran to completion on a small subset; without it, `cross.sh`, `correlation.sh` and `dependence.sh` fail.

| ID | Module | Conflict | Status |
|---|---|---|---|
| P1 | `llm/cross_information.py:13` | Reads metric columns by their old mcpp names (`V1`, `x1`, `s1`, …) and fails with `KeyError`. `metrics/calculate.py` has written paper names (`C5`, `C9`, `S1`, …) since `1babb9c` (2026-02-11). | [checked] proposal below, tested |
| P2 | `causal_analysis/dependence.py` | Reads old-name metric columns plus `idx` and `target` next to `logits`. Its input is the output of `llm/add_predictions.py`, which keeps only `input_ids`, `attention_mask`, `labels`, `logits`. | [checked] proposal below, tested. The column loss comes from `7b9bac1`: until then `add_predictions` loaded the dataset it was given and kept all columns. |
| P3 | `causal_analysis/correlation.py:81` | `mean_correlation(labels, …)` raises `ValueError`; until `7b9bac1` the call was `mean_correlation([labels], …)`. | [checked] proposal below, tested |
| P3b | `causal_analysis/correlation.py:56` | Loads metrics-classifier predictions from `hf-model/<slug>/autosklearn-model-10min/<uid>/checkpoint-*`, which nothing writes (`metrics/train.py:38` writes `hf-model/<slug>/<uid>/autosklearn-model-10min/predictions.json`), and raises. The loaded value is no longer used: before `7b9bac1` the classifier was a sixth entry ("scm") in the prediction-correlation matrix. | [checked] proposal below, tested; restoring the "scm" column is an author decision |
| P4 | `metrics/train.py:38` | Writes the metrics classifier into `hf-model/<slug>/<uid>/autosklearn-model-10min/`, next to the LLM checkpoints. `llm/aggregate_results.py:25`, `llm/eval_codegen.py`, `llm/eval_starcoder.py`, `llm/cross_information.py:21`, `causal_analysis/access.py` and `causal_analysis/cross.py` expect only `checkpoint-*` entries there and fail once it exists. No module aggregates these results (`metrics/aggregate.py` reads a different layout). | [checked for `aggregate_results`, `cross_information`; others audit] worked around in scripts, see below |

Script-side workaround for P4, no Python change: `scripts/run.sh` runs `metrics/train.sh` after all LLM and causal-analysis stages have ended, and `scripts/metrics/aggregate.sh` aggregates the results with `llm.aggregate_results` through a directory of symlinks (`<dataset>/autosklearn-view/<slug>/<uid>`). Consequence: re-running LLM evaluation or the causal stages after `metrics/train.sh` fails until the `autosklearn-model-10min` directories are moved away.

### Proposed patches

P1, one line, `src/llm/cross_information.py:13`. Uses the 23 columns `metrics.calculate` writes. The feature set equals the old one except that M3 is now `V5` instead of `V3` (decision 2); the column order changes, which a logistic regression on standardised features does not depend on beyond solver tolerance.

```diff
-relevant_metrics = ["C1", "C2", "C3", "C4", "V1", "V3", "V6", "V7", "V8", "V10", "x1", "x2", "x3", "x4", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "m1", "m2"]
+relevant_metrics = ["S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "M1", "M2", "M3", "T1", "T2", "T3"]
```

P3, restores the call as it was before `7b9bac1`, `src/causal_analysis/correlation.py:81`:

```diff
-            rs[f"{base_model}-true"] = mean_correlation(labels, [metrics[base_model][:, i]], False)
+            rs[f"{base_model}-true"] = mean_correlation([labels], [metrics[base_model][:, i]], False)
```

P3b, removes a load whose result is unused, `src/causal_analysis/correlation.py:56`. Output is unchanged because nothing reads `scm_predictions`:

```diff
-    scm_predictions = {base_model: load_all_predictions(models / model_slug(base_model) / "autosklearn-model-10min") for base_model in base_models}
```

P2, `src/causal_analysis/dependence.py` only (full diff in `docs/proposed-patches.diff`):

- Use the paper metric names directly; the old-name `MAPPING` (with M3 as `V3`) is removed. The order of `sorted_relevant_metrics` is unchanged.
- `load_all` joins each logits dataset column-wise with `data/primevul-lookaround/mcpp-microsoft__unixcoder-base-nine`, which carries the metrics, `idx` and `target`. This relies on both being derived from `primevul-lookaround/hf` without reordering, which holds for `metrics.calculate` and `llm.add_predictions`. The metrics are the UniXcoder-tokenised ones for all five models, because that is the only lookaround metrics dataset the scripts compute.
- The final summary concatenates three columns with `+`; with `datasets` ≥ 4 a column is no longer a list, so each is wrapped in `list(...)`. Same values.

Not changed by P2: the "StarCoder" entry still points at the CodeGen directory (decision 1).

## Choices made in the scripts that the authors should confirm

These are not in the Python code and were not recoverable from the repository.

- Slurm resources for jobs that had no header: `causal_analysis/cross.sh` (cpu-2d, 8 CPUs, 160 GB), `causal_analysis/dependence.sh` (cpu-2h, 4 CPUs, 40 GB), `vulnerability_evaluation/gnn.sh` (gpu-2d, 1 GPU, 4 CPUs, 40 GB).
- GNN baselines: Devign and ReVeal, seeds 0–9, default hyper-parameters of `graph.train`; Joern parsing and export run with half the allocated CPUs as parallel jobs.
- `metrics.calculate` is called with `--num-cpus` set to the allocated CPUs (64); the script did not pass it before, which means one process.
- `cpgtools.def` no longer installs `software-properties-common`: it is unused and its `systemd` dependency fails to configure in an unprivileged build.
- `cpgtools.def`: Joern v4.0.49 needs Java 17, but at run time `java` resolved to the system OpenJDK 11 (a dependency of the `gradle` package), because the pinned JDK 17.0.9 is installed under `/root/.sdkman` and was not on `PATH`. Every parse failed with `UnsupportedClassVersionError`. The pinned JDK is now copied to `/opt/java` and put on `PATH`. No Joern or Java version changed.
- `llm-requirements.txt`: `tree-sitter` was unpinned. With the current release (0.26) `mcpp` 1.3.4 raises inside `metrics.calculate`, which catches the error and writes 0 for every metric of every function, with exit code 0. Pinned to `tree-sitter==0.25.2` (the only minor version `mcpp` 1.3.4 works with), `tree-sitter-c==0.24.2`, `tree-sitter-cpp==0.23.4`. The grammar versions used for the paper are not recorded; they can change metric values. `scripts/metrics/calculate.sh` now fails if all metrics of a dataset are zero.
- `gnn-requirements.txt`: `transformers` 4.43.1 → 4.46.3. `graph/train.py:29` passes `num_items_in_batch` to `Trainer.compute_loss`, which exists from 4.46; with the pinned version training fails at the first step. 4.46.3 is the oldest version that works; the version the GNN code was written against is not recorded.
- `gnn-requirements.txt`: added `datasets==4.0.0`, `pyarrow==20.0.0`, `accelerate` and `pandas`, which the GNN modules import but the image did not contain. `datasets` 2.x cannot run `graph.datasets.primevul` (it passes a `Path` to `DatasetDict.save_to_disk`), and newer `pyarrow` needs NumPy 2 while the file pins a NumPy 1.x stack. The versions the authors used are not recorded.
- `graphs/preprocess.sh` stops if Joern exported no graph at all. Joern failures are otherwise only written to the report files.
- `llm-requirements.txt`: `datasets==5.0.1`, the version the unpinned build resolved to and the smoke test ran with. The supplied lookaround dataset uses the `List` feature type, which `datasets` < 4 cannot read.
- SAST: `sast.data`, `sast.scores` and `sast.classify` run in `llm.sif`. `build.sh` builds the tool images with `sast/apptainer_build.sh`. In the `sast` repository, `rats.def` and `cppcheck.def` no longer built: their Debian bullseye base is end-of-life and `apt-get update` fails. Both now use `debian:bookworm`, like `semgrep.def`; cppcheck's `libpcre++-dev`, which bookworm lacks, is replaced by `libpcre3-dev`. On 600 PrimeVul functions both tools give byte-identical output with the old and the new images. The change is in the public repository [mlsec-group/sast](https://github.com/mlsec-group/sast) (commit `49bd3da`, same tree as the tested local commit). `sast/` is a git submodule at that commit.
- `primevul-lookaround`: nothing in the repository creates it. `data/primevul-lookaround/hf` has to be supplied (splits `train`/`val`/`test`; columns `texts`, `labels`, `idx`, `target` with the four versions 1–4 per `idx`).

## Prompting experiment (GPT-3.5 Turbo, GPT-4o)

The code was not in the repository. It came as an archive from the author who ran it (`openai-experiment.zip`: scripts, the request files, the drawn subsets and the downloaded responses). It is now `src/prompting/`; data and responses are not committed.

| Archive | Now | Change |
|---|---|---|
| `create_raw_batches.py` | `prompting.raw_batches` | Paths are arguments: reads `<dataset>/hf` (the archive read a cluster path, `…/primevul/hf-full/hf/`), writes `<dataset>/prompting/raw-batches/`. The five sizes are still drawn in the original order from one `seed(1337)` stream. |
| `create_batches.py` | `prompting.batches` | Paths and model are arguments. |
| `submit.py` | `prompting.submit` | Paths, model and repetitions are arguments; the key is read from `OPENAI_API_KEY`. Stops with an error if a batch fails, expires or is cancelled (before: polled forever). |
| `classify.py` | `prompting.classify` | Paths and model are arguments; removed one unused assignment that was overwritten on the next line. |
| `openai_expr/utils.py` | `prompting.prompts` | Verbatim. |
| `openai_expr/run_prompting.py`, `run.sh` | not ported | Per-request path through the chat API on a file not in the archive (`primevul_test_paired_xs.jsonl`); not used for the paper's numbers. |

Checked on 2026-11-16 against the archive:

- `scripts/prompting/batches.sh` on `data/primevul/hf` built by `dataset/run.sh` from the PrimeVul release draws the same subsets as the archive (all five sizes, same functions in the same order; only the JSON key order differs) and writes request files identical to the archive's for both models.
- `scripts/prompting/classify.sh` on the archived responses (10 repetitions per model, 2,000 functions, 45 vulnerable) reproduces the paper's GPT rows except one value: GPT-3.5 F1 is 5.98 ± 0.83, the paper reports 5.80 ± 0.83. Every other GPT value matches to the printed precision, including the interval of that F1, and no subset of the ten repetitions gives 5.80 with that interval. Probably a transcription error in the paper.
- GPT-3.5 has 1,999 responses per repetition: request `hf-idx` 4896 (a benign function of 79,225 characters) exceeds its context window and is rejected by the API. GPT-4o has all 2,000.

Not run: `scripts/prompting/submit.sh`. It costs money, needs internet access and `OPENAI_API_KEY`, and `gpt-3.5-turbo-0125` and `gpt-4o-2024-08-06` may no longer be served. Responses are sampled with the API's default temperature, so a new run gives different numbers. `run.sh` runs this experiment only when named (`./scripts/run.sh prompting`). The `openai` version used originally is not recorded; `llm.sif` now has `openai==1.109.1`.

To check the scoring without the API, copy the archive's `data/out/` to `data/primevul/prompting/out/` and run `scripts/prompting/batches.sh` and `scripts/prompting/classify.sh`.

## Numbers reported in the paper

From the paper sources (`tables/classification_results_extended.tex`, `tex/04_evaluation.tex`). Mean ± 90 % confidence interval over 10 runs, in percent; the success criterion for the final run is the same ordering and similar magnitudes, not equal values (see the deviations above).

| Predictor | F1 | AUPRC | MCC | BAcc | VD-S ↓ | Script with the result |
|---|---|---|---|---|---|---|
| Code metrics classifier | 20.32 ± 0.59 | 13.80 ± 0.82 | 18.90 ± 0.55 | 62.22 ± 0.64 | 90.20 ± 0.70 | `metrics/aggregate.sh` |
| UniXcoder | 20.69 ± 1.43 | 13.32 ± 1.22 | 21.36 ± 1.06 | 67.23 ± 2.38 | 92.84 ± 0.79 | `llm_aggregate/unixcoder.sh` |
| PDBERT | 19.50 ± 1.17 | 13.38 ± 0.68 | 20.67 ± 0.40 | 69.19 ± 1.86 | 92.40 ± 1.23 | `llm_aggregate/pdbert.sh` |
| CodeGen 2.5 | 18.57 ± 0.54 | 13.22 ± 0.30 | 17.07 ± 0.41 | 63.23 ± 1.19 | 92.50 ± 0.49 | `llm_aggregate/codegen.sh` |
| LineVul (CodeBERT) | 18.48 ± 1.23 | 11.68 ± 0.82 | 19.34 ± 0.93 | 68.36 ± 2.06 | 92.66 ± 1.35 | `llm_aggregate/codebert.sh` |
| StarCoder 2 | 17.09 ± 0.41 | 12.46 ± 0.40 | 19.73 ± 0.48 | 72.25 ± 1.41 | 94.17 ± 0.44 | `llm_aggregate/starcoder.sh` |
| GPT-3.5 Turbo | 5.80 ± 0.83 | 2.56 ± 0.18 | 3.34 ± 1.01 | 52.96 ± 1.09 | 100.00 ± 0.00 | `prompting/classify.sh` |
| GPT-4o | 5.31 ± 0.35 | 2.50 ± 0.10 | 2.72 ± 0.90 | 54.17 ± 1.47 | 100.00 ± 0.00 | `prompting/classify.sh` |
| Random | 4.42 ± 0.19 | 2.33 ± 0.01 | 0.05 ± 0.36 | 50.15 ± 0.45 | 99.60 ± 0.18 | none |
| GNN (best of Devign, ReVeal) | 17.8 ± 3.85 | | | | | `gnn_aggregate.sh` |
| SAST (best of RATS, Semgrep) | 14.2 | | | | | `sast.sh` |

Not produced by any code in the repository: the Random row and the table's "Efficiency" column. The paper's tables on metrics from embeddings and metric correlations (`tables/metrics_from_embeddings.tex`, `tables/metric_correlations.tex`) still label metrics with the old mcpp names (`x3`, `t2`, `V3`, …), while the code now prints the paper names.

## Smoke test (2026-11-13, CPU only)

Run on subsets of PrimeVul (80 to 1,300 functions) and of the lookaround dataset, outside the repository.

| Stage | Result |
|---|---|
| `build.sh` images (`llm`, `metrics`, `cpgtools`, `gnn`, SAST tools) | built |
| `dataset/run.sh`, `metrics/calculate.sh` (UniXcoder, CodeBERT, lookaround) | ran |
| `metrics/isolation.sh`, `leaveoneout.sh` (one task each) | ran |
| `metrics/train.sh` (UniXcoder, CodeBERT, one run), `metrics/aggregate.sh` | ran, including the symlink view for the classifier results |
| `llm_finetune/unixcoder.sh` | starts and trains; not run to a checkpoint (CPU) |
| `llm_eval`, `llm_aggregate` (UniXcoder, CodeBERT) | ran on an un-finetuned model saved as a stand-in checkpoint |
| `graphs/preprocess.sh` | ran |
| `gnn.sh` (Devign, one seed, 938 training graphs), `gnn_aggregate.sh` | ran (2.4 h on CPU). The Trainer logged "Could not locate the best model" at the end of training (decision 9). |
| `prompting/batches.sh`, `prompting/classify.sh` (2026-11-16) | ran on the full PrimeVul test split and the archived responses; see [Prompting experiment](#prompting-experiment-gpt-35-turbo-gpt-4o) |
| `sast.sh` | ran |
| `add_predictions` (both datasets), `add_embeddings`, `access_per_model` (UniXcoder) | ran on the stand-in checkpoint |
| `access.sh`, `cross.sh`, `cross_aggregate.sh`, `correlation.sh`, `dependence.sh` | ran with the proposed patches; the five models and eleven run ids were copies of the one UniXcoder run |

Not verified: anything with CodeGen, StarCoder or PDBERT; finetuning to a real checkpoint; GPU execution; Slurm submission (only the printed `sbatch` plan was reviewed).


## Approved Python patches

Approved by an author on 2026-11-16 and listed in `scripts/dev/approved_patches.txt`.

| ID | File | Change |
|---|---|---|
| A1 = P1 | `llm/cross_information.py` | Paper metric names, as proposed. |
| A2 = P3, P3b | `causal_analysis/correlation.py` | `mean_correlation([labels], …)`; unused load of `autosklearn-model-10min` predictions removed; no "scm" column, as in the baseline. |
| A3 = P2 | `causal_analysis/dependence.py` | As proposed, except that each model's logits are joined with that model's lookaround metrics, `primevul-lookaround/mcpp-<slug>`, not with the UniXcoder ones. `scripts/metrics/calculate.sh` now computes the lookaround metrics in every array task (before: task 0 only). |
| A4 | `causal_analysis/access.py`, `causal_analysis/correlation.py` | Read runs `0..9` (decision 6). |
| A5 | `causal_analysis/dependence.py` | "StarCoder" points at `bigcode__starcoder2-7b` (decision 1). |
| A6 | `llm/finetune_codegen.py`, `llm/finetune_starcoder.py` | Train batch layout 4×16 → 8×8 (per-device batch 8, gradient accumulation 8), as in the paper's table; effective batch size unchanged at 64. Per-device memory doubles; the GPU allocations in `llm_finetune/{codegen,starcoder}.sh` are unchanged. |
