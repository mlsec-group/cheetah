![Logo](logo.png)

# LLM-based Vulnerability Discovery through the Lens of Code Metrics

This code belongs to the submission
> Anonymous authors. 2025. LLM-based Vulnerability Discovery through the Lens of Code Metrics.

## Overview
In our submission we analyze the performance of large language models for vulnerability discovery through the lens of simple code metrics. We find that a classifier trained solely on code metrics performs on par with the approaches based on LLMs. A root-cause analysis further reveals a strong correlation and a causal effect between LLMs and code metrics: When the value of a metric is changed, LLM predictions tend to shift by a corresponding magnitude. This dependency suggests that LLMs operate at a similarly shallow level as code metrics, limiting their ability to grasp complex patterns and fully realize their potential in vulnerability discovery. In this repository, we share the code to reproduce our experiments.

## Dependencies
- [Apptainer](https://apptainer.org/) (or Singularity) for containerized runs
- A CUDA-capable host for LLM finetuning/evaluation and GNN training (CPU-only steps use separate images where noted)
- Bash; Hugging Face cache configuration via `huggingface.env` (see `scripts/config.sh`)

## Environment layout
Experiments assume the **repository root is the current working directory** (interactive sessions and Slurm jobs). Shell helpers invoke Apptainer with `PYTHONPATH="${PWD}/src"` so the Python packages under `src/` are used **without** installing the repo into each image. The project tree is visible inside the container at the **same paths as on the host** (Apptainer’s default behavior for the process working directory).

Several **`.def` / `.sif` pairs** cover different stacks, e.g. `container.sif` (LLM + metrics + causal helpers), `auto.sif` (Auto-sklearn metric experiments), `dataset.sif` (light CPU), `gnn-preprocessing.sif` (Joern/CPG), and `gnn-training.sif` (PyTorch/PyG). Build them from the repo root (see [Reproducibility](#reproducibility) for the full list and variables such as `GNN_PREPROCESS_SIF` / `GNN_TRAINING_SIF`).

Individual steps are run as **`python -m <module> ...`** (for example `python -m llm.finetune_bertlike`, `python -m metrics.calculate`). The wrapper `scripts/apptainer_cheetah.sh` loads `scripts/lib/run_in_sif.sh` and defines `_cheetah_run` for use across `scripts/`.

## Quick start
To build the Apptainer images and run the full scripted pipeline:
```
./scripts/build.sh
./scripts/run.sh
```
The build script in this repository may only compile a **subset** of images by default; enable or add `apptainer build …` lines for every SIF you need (see [docs/reproducibility.md](docs/reproducibility.md)). Configure tokens and cache paths in `huggingface.env` before LLM or dataset steps.

## Reproducibility
For **detailed** instructions—image build commands, which `python -m` modules belong to which SIF, Hugging Face and Slurm conventions, and a high-level script dependency graph—see **[docs/reproducibility.md](docs/reproducibility.md)**.

## Dataset
We use [PrimeVul](https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00038) as the basis for our experiments. Before running individual stages, obtain the dataset and place it where the scripts expect it (the GNN and preprocessing examples use `data/primevul/raw` with JSONL splits). If you use the automated download path in `scripts/build.sh`, ensure filenames and layout match what `graph.datasets.primevul preprocess` expects, or adapt paths in your copy of the scripts.

## Experiments
We conduct three experiments in our paper: A comparative assessment of vulnerability discovery approaches, a performance analysis of code metrics for vulnerability discovery, and an analysis of LLM-based discovery approaches through the lens of code metrics. Each of these experiments aligns with one sub-directory in the `scripts` directory. Before conducting any of the experiments, build the relevant Apptainer images, configure the runtime environment, and prepare PrimeVul (see [Dataset](#dataset) and [Reproducibility](#reproducibility)).

#### Comparative analysis of vulnerability discovery approaches
In our comparative analysis of vulnerability discovery approaches we consider five state-of-the-art LLM-based methods, two graph neural network (GNN) approaches, two static application security testing (SAST) tools and a classifier trained on our 23 syntactic code metrics. The directory `scripts/vulnerability_evaluation` contains the code to train and evaluate these methods on the [PrimeVul](https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00038) dataset.

To conduct this experiment, run:
```
./scripts/vulnerability_evaluation/run.sh
```

#### Performance analysis of code metrics for vulnerability discovery
We assess the predictive performance of the 23 considered code metrics in a second experiment. To that end, we train a classifier on each metric individually to assess its performance in isolation and train a classifier on every metric but this one, to assess the impact of its absence on the quality of the classifier. The code to start this experiment is contained in the `scripts/metric_analysis` directory.

The experiment can be started by executing:
```
./scripts/metric_analysis/run.sh
```

#### Analysis of LLM-based vulnerability discovery through the lens of code metrics
Given the high interpretability of the considered code metrics, we use them as a tool to gain insights into the decision process of LLM-based vulnerability discovery approaches in our last experiment. Specifically, we conduct four tests:
1. _Information access._ We check whether the embeddings created by the language models that are used as inputs to the succeeding classification heads contain information about the code metrics.
2. _Information correlation._ We assess whether the decision with the vulnerability discovery models correlates with the code metric values.
3. _Information dependence._ We further turn to an investigation of joint changes in code metrics and prediction probability as a strong indicator for causality.
4. _Cross information._ Lastly, explicitly augmenting the language model based classifiers with the code metric features tests whether any information orthogonal to code metrics are being used.
This experiment is located in the `scripts/causal_analysis` directory.

To run the experiment, execute:
```
./scripts/causal_analysis/run.sh
```
