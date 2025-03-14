![Logo](logo.png)

# LLM-based Vulnerability Discovery through the Lens of Code Metrics

This code belongs to the submission
> Anonymous authors. 2025. LLM-based Vulnerability Discovery through the Lens of Code Metrics.

## Overview
In our submission we analyze the performance of large language models for vulnerability discovery through the lens of simple code metrics. We find that a classifier trained solely on code metrics performs on par with the approaches based on LLMs. A root-cause analysis further reveals a strong correlation and a causal effect between LLms and code metrics: When the value of a metric is changed, LLM predictions tend to shift by a corresponding magnitude. This dependency suggests that LLMs operate at a similarly shallow level as code metrics, limiting their ability to grasp complex patterns and fully realize their potential in vulnerability discovery. In this repository, we share the code to reproduce our experiments.

## Dependencies
- apptainer

## Quick start
To build the apptainer container and conduct all experiments from the paper, run:
```
./scripts/build.sh
./scripts/run.sh
```

## Structure
Each of our three experiments in the paper aligns with one sub-directory in the `scripts` directory.

#### Comparative analysis of vulnerability discovery approaches
In our comparative analysis of vulnerability discovery approaches we consider five state-of-the-art LLM-based methods, two graph neural network (GNN) approaches, two static application security testing (SAST) tools and a classifier trained on our 23 syntactic code metrics. The directory `scripts/vulnerability_evaluation` contains the code to train and evaluate these methods on the [PrimeVul](https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00038) dataset.

To conduct this experiment, run:
```
./scripts/vulnerability_evaluation/run.sh
```

#### Performance analysis of code metrics for vulnerability discovery


#### Analysis of LLM-based vulnerability discovery through the lens of code metrics

## Dataset
For our dataset 
