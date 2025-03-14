import json
from pathlib import Path
import os

import torch
from loguru import logger as log
from omegaconf import OmegaConf
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
from optuna import Trial

from gnn-baselines.config.schema import HyperParameters, FixedParameters, Experiment, ModelConfig, Dataset as DatasetConfig
from gnn-baselines.datasets import load_dataset
from gnn-baselines.models import init_model, get_feature_set
from gnn-baselines.config.cmd import Split

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision('medium')

DEBUG_RUN = False
if DEBUG_RUN:
    LIMIT_TRAINING_BATCHES = 0.1
    LIMIT_VAL_BATCHES = 100
else:
    LIMIT_TRAINING_BATCHES = 1.0
    LIMIT_VAL_BATCHES = 1.0


def setup_training(
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
    test_workers: int,
    model_override_kwargs=None,
    override_callbacks=None,
    override_trainer_kwargs=None,
    override_optim_kwargs=None,
    override_data_kwargs=None,
    override_experiment_kwargs=None,
    disable_logger=False,
    limit_training_batches=LIMIT_TRAINING_BATCHES,
    limit_val_batches=LIMIT_VAL_BATCHES,
    test_predictions_path: Path = None
):
    dataset_name = experiment_cfg.dataset_name
    seed_everything(experiment_cfg.seed)
    train_workers = min(train_workers, os.cpu_count())
    val_workers = min(val_workers, os.cpu_count())
    test_workers = min(test_workers, os.cpu_count())

    batch_size = hyperparams.batch_size
    if override_data_kwargs is not None and "batch_size" in override_data_kwargs:
        batch_size = override_data_kwargs["batch_size"]
    ds_module = load_dataset(
        dataset_name,
        split_cfg,
        batch_size,
        train_workers,
        val_workers,
        test_workers,
        experiment_cfg.seed,
        root=dataset_cfg.root,
        feature_set=get_feature_set(model_name),
        processed_file=dataset_cfg.processed_file,
        node_type_vocab=dataset_cfg.node_type_vocab,
        w2v_kv_file=dataset_cfg.w2v_kv_file,
        tokenizer_cfg=dataset_cfg.tokenizer_cfg,
        interim_dir=dataset_cfg.interim_dir,
        jobs=dataset_cfg.jobs
    )

    # import numpy as np
    # X_train, Y_train = [], []
    # for batch in ds_module.train_dataloader():
    #     feat = [batch[k].numpy() for k in batch.keys() if k.startswith("mcpp_")]
    #     X_train.append(np.array(feat).T)
    #     Y_train.append(batch.y.numpy())
    # X_train = np.vstack(X_train)
    # Y_train = np.concatenate(Y_train)
    # X_val, Y_val = [], []
    # for batch in ds_module.val_dataloader():
    #     feat = [batch[k].numpy() for k in batch.keys() if k.startswith("mcpp_")]
    #     X_val.append(np.array(feat).T)
    #     Y_val.append(batch.y.numpy())
    # X_val = np.vstack(X_val)
    # Y_val = np.concatenate(Y_val)
    # X_test, Y_test = [], []
    # for batch in ds_module.test_dataloader():
    #     feat = [batch[k].numpy() for k in batch.keys() if k.startswith("mcpp_")]
    #     X_test.append(np.array(feat).T)
    #     Y_test.append(batch.y.numpy())
    # X_test = np.vstack(X_test)
    # Y_test = np.concatenate(Y_test)
    # np.savez("mcpp.data.npz", X_train, Y_train, X_val, Y_val, X_test, Y_test)
    # exit(0)
    model_kwargs = OmegaConf.to_container(model_cfg[model_name], resolve=True)
    extra_model_kwargs = {}
    if test_predictions_path is not None:
        extra_model_kwargs["test_predictions_path"] = test_predictions_path
    # TODO: just a hotfix for integrating linear MCPP
    if model_name == "linear_mcpp":
        mean = ds_module.dataset.stats["mean"]
        std = ds_module.dataset.stats["std"]
        # mean, std = ds_stats(ds_module.dataset, model_cfg["linear_mcpp"].mcpp_features)
        extra_model_kwargs["mean"] = mean
        extra_model_kwargs["std"] = std
    if model_name == "devign":
        extra_model_kwargs["type_dim"] = ds_module.dataset.node_type_embedding.dims

    model_kwargs.update(extra_model_kwargs)
    if model_override_kwargs is not None:
        model_kwargs.update(model_override_kwargs)

    optim_kwargs = {
        "lr": hyperparams.lr,
        "weight_decay": hyperparams.weight_decay,
        "amsgrad": True
    }
    if override_optim_kwargs is not None:
        optim_kwargs.update(override_optim_kwargs)

    experiment_kwargs = {
        "use_smote": experiment_cfg.use_smote,
        "use_triplet_loss": experiment_cfg.use_triplet_loss,
        "triplet_alpha": hyperparams.triplet_alpha,
        "triplet_beta": hyperparams.triplet_beta,
        "triplet_gamma": hyperparams.triplet_gamma,
    }
    if override_experiment_kwargs is not None:
        experiment_kwargs.update(override_experiment_kwargs)

    model = init_model(
        model_name,
        ds_module.dataset.stats["class_ratio"],
        **model_kwargs,
        **optim_kwargs,
        **experiment_kwargs,
    )

    # experiment_tag = "hyper-smote"
    experiment_tag = "no_smote-no_triplet_loss"
    best_name = f"{model.name}-best"
    best_ckpt = model_dir / experiment_tag / f"{best_name}.ckpt"
    best_ckpt.parent.mkdir(exist_ok=True, parents=True)
    if disable_logger:
        logger = None
    else:
        logger = TensorBoardLogger("tb_logs", name=experiment_tag)

    if override_callbacks is None:
        callbacks = [
            ModelCheckpoint(model_dir / experiment_tag, best_name, monitor="val/f1_score", mode="max"),
            EarlyStopping(monitor="val/f1_score", mode="max", patience=fixed_params.patience),
            RichModelSummary(max_depth=3),
            RichProgressBar()
        ]
    else:
        callbacks = override_callbacks

    trainer_kwargs = {
        "accumulate_grad_batches": hyperparams.accumulated_batches
    }
    if override_trainer_kwargs is not None:
        trainer_kwargs.update(override_trainer_kwargs)

    trainer = Trainer(
        accelerator="auto",
        max_epochs=fixed_params.max_epochs,
        log_every_n_steps=50,
        # profiler="simple",
        limit_train_batches=limit_training_batches,
        limit_val_batches=limit_val_batches,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        **trainer_kwargs
    )
    return model, ds_module, trainer, best_ckpt


def train(
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
    test_workers: int,
    test_predictions_path: Path
):
    model, ds_module, trainer, best_ckpt = setup_training(
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
        test_workers,
        test_predictions_path=test_predictions_path
    )
    if fixed_params.max_epochs == 0:
        log.info("Skipping training")
        exit(0)
    trainer.fit(model, datamodule=ds_module)
    train_scores = trainer.callback_metrics
    test_scores = trainer.test(model, datamodule=ds_module, ckpt_path=best_ckpt)
    if isinstance(test_scores, list):
        test_scores = test_scores[0]
    with open(model_dir / "scores.json", "w") as f:
        json.dump({
            "train": {key: val.item() if isinstance(val, torch.Tensor) else val for key, val in train_scores.items()},
            "test": {key: val.item() if isinstance(val, torch.Tensor) else val for key, val in test_scores.items()}
        }, f, indent=4)


def ds_stats(dataset, mcpp_features):
    vecs = []
    mcpp_features = [f"mcpp_{feat}" for feat in mcpp_features]
    for data in dataset:
        feat = [data[f] for f in mcpp_features]
        vecs.append(feat)
    vecs = torch.tensor(vecs)
    return vecs.mean(dim=0), vecs.std(dim=0)


def suggest_trainer_hparams(trial: Trial):
    return {
        "accumulate_grad_batches": trial.suggest_int("hparams.accumulated_batches", 1, 64)
    }


def suggest_experiment_hparams(trial: Trial):
    # hparams = {"use_smote": trial.suggest_categorical("experiment.use_smote", [True, False])}
    # use_triplet = trial.suggest_categorical("experiment.use_triplet_loss", [True, False])
    use_triplet = False
    hparams = {"use_smote": False}
    if use_triplet:
        hparams.update({
            "use_triplet_loss": use_triplet,
            "triplet_alpha": trial.suggest_float("hparams.triplet_alpha", 0, 1),
            "triplet_beta": trial.suggest_float("hparams.triplet_beta", 0, 1),
            "triplet_gamma": trial.suggest_float("hparams.triplet_gamma", 0, 1),
        })
    return hparams
