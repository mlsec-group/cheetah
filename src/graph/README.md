# GNN Baselines

Code for the GNN baselines in the cheetah research project

## Setup

- build docker container
- mount primevul dataset directory to `/gnn-baselines/data/primevul`
- mount source directory to `/gnn-baselines/src/gnn-baselines`

## Data Extraction

- run commands in `run_all.sh`

## Training

Example: train unixcoder on a random split

```
gnn-baselines train start experiment.model_name=unixcoder
```

Launch tensorboard to observe progress:

```
tensorboard --logdir tb_logs
```

## Hyperparameter Optimization

- start postgres server: `service postgresql start`
- start optuna dashboard: `optuna-dashboard postgresql://dbuser:password@localhost:5432/gnn-baselines`
- run `python experiments/01b_reset_study.py $model` to create (or reset) a study (only once)
- run `CUDA_VISIBLE_DEVICES=<some_device> python experiments/01_hparam_tuning.py devign experiments/results/devign-study.json`
    - you can start it multiple times to increase the number of trials (in parallel)
- visit `htpp://localhost:8080` to observe the progress


## Evaluation

The eval is currently implemented in `experiments/02_train.py`. It takes the best params found in `experiments/results/$model-study.json` and trains the model on the complete dataset.

The output is a `$model-eval.json` containing all the relevant metrics.

Note: The test loader tries to load the best model in `data/models/$study-name/$model-best.ckpt`. Make sure to delete this if there is a file from a previous run.

Example:

```sh
device=3
model=unixcoder
CUDA_VISIBLE_DEVICES=$device python experiments/02_evaluation.py $model experiments/results/$model-study.json experiments/results/$model-eval.json
```
