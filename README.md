![Logo](logo.png)

# LLM-based Vulnerability Discovery through the Lens of Code Metrics

This code belongs to the submission
> Anonymous authors. 2025. LLM-based Vulnerability Discovery through the Lens of Code Metrics.

## Overview
In our submission we analyze the performance of large language models for vulnerability discovery through the lens of simple code metrics. We find that a classifier trained solely on code metrics performs on par with the approaches based on LLMs. A root-cause analysis further reveals a strong correlation and a causal effect between LLms and code metrics: When the value of a metric is changed, LLM predictions tend to shift by a corresponding magnitude. This dependency suggests that LLMs operate at a similarly shallow level as code metrics, limiting their ability to grasp complex patterns and fully realize their potential in vulnerability discovery. In this repository, we share the code to reproduce our experiments.

## Dependencies
- apptainer

## Quick start
To setup the environment and conduct all experiments from the paper, run:
```
./scripts/build.sh
./scripts/run.sh
```

## Dataset
We resort to [PrimeVul](https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00038) as a basis to conduct our experiments on. Prior to executing any of the experiments individually, make sure that you downloaded the PrimeVul dataset and placed it in the `data/raw` directory. If you followed the [Quick start](#quick-start) section, this step is not necessary as the dataset was already downloaded automatically in this case.

## Experiments
We conduct three experiments in our paper: A comparative assessment of vulnerability discovery approaches, a performance analysis of code metrics for vulnerability discovery, and an analysis of LLM-based discovery approaches through the lens of code metrics. Each of these experiments aligns with one sub-directory in the `scripts` directory. Before conducting any of the experiments, make sure that the environment was setup by creating the apptainer container through `./scripts/build.sh` and that the PrimeVul dataset is downloaded and preprocessed (see [Dataset](#dataset)).

#### Comparative analysis of vulnerability discovery approaches
In our comparative analysis of vulnerability discovery approaches we consider five state-of-the-art LLM-based methods, two graph neural network (GNN) approaches, two static application security testing (SAST) tools and a classifier trained on our 23 syntactic code metrics. The directory `scripts/vulnerability_evaluation` contains the code to train and evaluate these methods on the [PrimeVul](https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00038) dataset.

To conduct this experiment, run:
```
./scripts/vulnerability_evaluation/run.sh
```

#### Performance analysis of code metrics for vulnerability discovery
We assess the predictive performance of the 23 considered code metrics in a second experiment. To that end, we train a classifier on each metric individually to assess its performance in isolation and train a classifier on every metric but this one, to assess the impact of its absence on the quality of the classifier. The code to start this experiment is contained in the `scripts/metric_analysis` directory.

The experiment can be started, by executing:
```
./scripts/metric_analysis/run.sh
```

#### Analysis of LLM-based vulnerability discovery through the lens of code metrics
Given the high interpretability of the considered code metrics, we use them as a tool to gain a insights into the decision process of LLM-based vulnerability discovery approaches in our last experiment. Specifically, we conduct four tests:
1. **Information access.** We check whether the embeddings created by the language models that are used as inputs to the succeeding classification heads contain information about the code metrics.
2. **Information correlation.** We assess whether the decision with the vulnerability discovery models correlates with the code metric values.
3. **Information dependence.** We further turn to an investigation of joint changes in code metrics and prediction probability as a strong indicator for causality.
4. **Cross information.** Lastly, explicitly augmenting the language model based classifiers with the code metric features tests whether any information orthogonal to code metrics are being used.
This experiment is located in the `scripts/causal_analysis` directory.

To run the experiment, execute:
```
./scripts/causal_analysis/run.sh
```
