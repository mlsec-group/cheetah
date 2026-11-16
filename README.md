![Logo](logo.png)

# LLM-based Vulnerability Discovery through the Lens of Code Metrics

This code belongs to the submission
> Anonymous authors. 2025. LLM-based Vulnerability Discovery through the Lens of Code Metrics.

## Overview
In our submission we analyze the performance of large language models for vulnerability discovery through the lens of simple code metrics. We find that a classifier trained solely on code metrics performs on par with the approaches based on LLMs. A root-cause analysis further reveals a strong correlation and a causal effect between LLMs and code metrics: When the value of a metric is changed, LLM predictions tend to shift by a corresponding magnitude. This dependency suggests that LLMs operate at a similarly shallow level as code metrics, limiting their ability to grasp complex patterns and fully realize their potential in vulnerability discovery. In this repository, we share the code to reproduce our experiments.

## Dependencies
- [Apptainer](https://apptainer.org/) (or Singularity) for containerized runs
- Slurm for the full experiments; a CUDA-capable GPU for LLM finetuning/evaluation and GNN training
- Bash; Hugging Face cache configuration via `huggingface.env`

## Environment layout
All scripts assume the **repository root is the current working directory**. Every step runs as `python -m <module> ...` inside an Apptainer image with `PYTHONPATH="${PWD}/src"`; the repository is never installed into an image. `scripts/config.sh` holds the shared settings (paths, images, the list of models).

Four images are built into `.sif/`: `llm.sif` (LLMs, metric extraction, causal analysis), `metrics.sif` (auto-sklearn metric classifiers), `cpgtools.sif` (Joern code property graphs) and `gnn.sif` (GNN training).

## Quick start
```
git submodule update --init   # SAST tools (or clone with --recurse-submodules)
./scripts/build.sh   # images, PrimeVul, PDBERT; needs internet access
./scripts/run.sh     # submits every stage to Slurm with job dependencies
```
`./scripts/run.sh <experiment>...` runs only the named experiments (`dataset`, `vulnerability_evaluation`, `metrics`, `causal_analysis`, and the opt-in `prompting`). `slurm=0 ./scripts/run.sh` runs the stages in the current shell, and `DRY_RUN=1 ./scripts/run.sh` prints what would be submitted.

## Reproducibility
To run on a different Slurm cluster, see **[CLUSTER_CONFIG.md](CLUSTER_CONFIG.md)** for every script and setting to adapt. See **[docs/reproducibility.md](docs/reproducibility.md)** for the stage graph, the image used by each module and the Slurm conventions, and **[docs/reproduction-report.md](docs/reproduction-report.md)** for known deviations from the setup used in the paper and open issues.

## Dataset
We use [PrimeVul](https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00038) as the basis for our experiments. `scripts/build.sh` downloads the three splits to `data/primevul/raw/primevul_{train,valid,test}.jsonl`.

## Experiments
We conduct three experiments in our paper: A comparative assessment of vulnerability discovery approaches, a performance analysis of code metrics for vulnerability discovery, and an analysis of LLM-based discovery approaches through the lens of code metrics. Each of these experiments aligns with one sub-directory in the `scripts` directory. Before conducting any of the experiments, run `./scripts/build.sh` and `./scripts/run.sh dataset`.

#### Comparative analysis of vulnerability discovery approaches
In our comparative analysis of vulnerability discovery approaches we consider five state-of-the-art LLM-based methods, two graph neural network (GNN) approaches, two static application security testing (SAST) tools and a classifier trained on our 23 syntactic code metrics. The directory `scripts/vulnerability_evaluation` contains the code to train and evaluate these methods on the [PrimeVul](https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00038) dataset.

To conduct this experiment, run:
```
./scripts/run.sh vulnerability_evaluation
```

The comparison also prompts GPT-3.5 Turbo and GPT-4o through the OpenAI Batch API on a stratified subset of 2,000 test functions (`scripts/prompting`). This is paid and is not part of a plain `./scripts/run.sh`:
```
OPENAI_API_KEY=... ./scripts/run.sh prompting
```

#### Performance analysis of code metrics for vulnerability discovery
We assess the predictive performance of the 23 considered code metrics in a second experiment. To that end, we train a classifier on each metric individually to assess its performance in isolation and train a classifier on every metric but this one, to assess the impact of its absence on the quality of the classifier. The code to start this experiment is contained in the `scripts/metrics` directory.

The experiment can be started by executing:
```
./scripts/run.sh metrics
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
./scripts/run.sh causal_analysis
```
