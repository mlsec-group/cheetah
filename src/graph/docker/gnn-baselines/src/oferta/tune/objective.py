from pathlib import Path

import optuna
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar

from gnn-baselines.train.callbacks import PyTorchLightningPruningCallback
from gnn-baselines.models import MODELS
from gnn-baselines.datasets.base import DataModule
from gnn-baselines.train.trainer import setup_training, suggest_trainer_hparams, suggest_experiment_hparams
from gnn-baselines.config.schema import HyperParameters, FixedParameters, Experiment, ModelConfig, Dataset as DatasetConfig
from gnn-baselines.config.cmd import Split

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")


TRAIN_BATCHES = 0.1
VAL_BATCHES = 100


def get_objective(
    model_name: str,
    model_dir: Path,
    model_cfg: ModelConfig,
    dataset_cfg: DatasetConfig,
    experiment_cfg: Experiment,
    hyperparams: HyperParameters,
    fixed_params: FixedParameters,
    split_cfg: Split,
    train_workers: int,
    val_workers: int,
    monitor: str = "val/loss",
    direction: str = "minimize"
):
    def objective(trial: optuna.trial.Trial):
        mode = "min" if direction == "minimize" else "max"
        model_cls = MODELS[model_name]
        model_kwargs = model_cls.suggest_hparams(trial)
        optim_kwargs = model_cls.suggest_optim_hparams(trial)
        experiment_kwargs = suggest_experiment_hparams(trial)
        data_kwargs = DataModule.suggest_data_hparams(trial)
        trainer_kwargs = suggest_trainer_hparams(trial)

        callbacks = [
            PyTorchLightningPruningCallback(trial, monitor=monitor),
            EarlyStopping(monitor=monitor, mode=mode, patience=fixed_params.patience),
            RichProgressBar()
        ]
        model, ds_module, trainer, _ = setup_training(
            model_name,
            model_dir,
            model_cfg,
            dataset_cfg,
            experiment_cfg,
            hyperparams,
            fixed_params,
            split_cfg,
            train_workers,
            val_workers,
            0,
            model_kwargs,
            callbacks,
            disable_logger=True,
            limit_training_batches=TRAIN_BATCHES,
            limit_val_batches=VAL_BATCHES,
            override_trainer_kwargs=trainer_kwargs,
            override_optim_kwargs=optim_kwargs,
            override_data_kwargs=data_kwargs,
            override_experiment_kwargs=experiment_kwargs,
            test_predictions_path=-1
        )

        trainer.fit(model, datamodule=ds_module)
        for key, val in trainer.logged_metrics.items():
            trial.set_user_attr(key, val.item())
        return trainer.logged_metrics[monitor].item()
    return objective
