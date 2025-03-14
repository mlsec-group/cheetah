from abc import ABC, abstractmethod, abstractstaticmethod
import json
from pathlib import Path
from dataclasses import dataclass
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from lightning import LightningModule
from torch_geometric.data import Data
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryF1Score
from optuna import Trial
from loguru import logger

from gnn-baselines.train.smote import smote, projection_loss, regularization_loss
from gnn-baselines.train.metrics import BalancedAccuracy, PositiveRate, VDScore


import warnings


@dataclass
class MetricLog:
    loss: float
    prec: float
    rec: float
    f1_score: float
    balanced_acc: float
    pos_rate: float
    vd_score: float

    def log(self, log_fn, prefix, training=False, prog_bar=False, log_kwargs=None):
        if log_kwargs is None:
            log_kwargs = {}
        log_fn(f"{prefix}/loss", self.loss, prog_bar=prog_bar, **log_kwargs)
        log_fn(f"{prefix}/balanced_accuracy", self.balanced_acc, prog_bar=prog_bar, **log_kwargs)
        log_fn(f"{prefix}/positive_rate", self.pos_rate, prog_bar=prog_bar, **log_kwargs)
        if not training:
            log_fn(f"{prefix}/precision", self.prec, prog_bar=False, **log_kwargs)
            log_fn(f"{prefix}/recall", self.rec, prog_bar=False, **log_kwargs)
            log_fn(f"{prefix}/f1_score", self.f1_score, prog_bar=False, **log_kwargs)
            log_fn(f"{prefix}/vd_score", self.vd_score, prog_bar=False, **log_kwargs)


def write_predictions(scores, preds, targets, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=["score", "pred", "target"])
        writer.writeheader()
        for score, pred, target in zip(scores, preds, targets):
            score = score.item()
            pred = pred.item()
            target = target.item()
            writer.writerow(dict(score=score, pred=pred, target=target))


class Model(ABC, LightningModule):
    def __init__(self, name, feature_set, class_ratio, test_predictions_path, **hparams):
        super().__init__()
        self.name = name
        self.feature_set = feature_set
        self.class_ratio = class_ratio
        self.class_weights = 1 - class_ratio
        self.out_dim = 1
        # self.out_dim = class_ratio.numel()
        self.test_predictions_path = test_predictions_path
        hparam_keys = (
            "use_smote",
            "use_triplet_loss",
            "lr",
            "weight_decay",
            "triplet_alpha",
            "triplet_beta",
            "triplet_gamma",
        )
        for key in hparam_keys:
            if key not in hparams:
                raise ValueError(f"Missing hyperparameter: {key}")
        self.save_hyperparameters(hparams)

        self.balanced_train_acc = BalancedAccuracy()
        self.train_pos_rate = PositiveRate()
        self.precision = BinaryPrecision(zero_division=0)
        self.recall = BinaryRecall(zero_division=0)
        self.f1 = BinaryF1Score(zero_division=0)
        self.balanced_acc = BalancedAccuracy()
        self.pos_rate = PositiveRate()
        self.vd_score = VDScore(target_fpr=0.005)

    @property
    @abstractmethod
    def embedder(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def classifier(self):
        raise NotImplementedError()

    @staticmethod
    @abstractstaticmethod
    def suggest_hparams(trial: Trial):
        raise NotImplementedError()

    def forward_embedding(self, data: Data):
        return self.embedder(data)

    def forward(self, data: Data):
        emb = self.forward_embedding(data)
        # output = F.softmax(self.classifier(emb), dim=1) # We are using binary_cross_entropy_with_logits now
        output = self.classifier(emb)
        return output

    def step(self, data: Data, batch_idx: int, eval_metrics=False):
        self.class_weights = self.class_weights.to(data.y.device)
        batch_class_weights = self.class_weights[data.y]
        target = data.y

        emb = self.forward_embedding(data)
        if self.hparams.use_smote and self.training:
            emb, target = smote(emb, target)
            if target.shape[0] == 0:
                logger.warning(f"SMOTE created an empty batch (idx={batch_idx}, y_pos_rate={data.y.float().mean()})")
                return 0
            batch_class_weights = torch.ones_like(target)

        emb_penultimate = self.classifier[:-1](emb)
        output = self.classifier[-1](emb_penultimate).flatten()

        loss = F.binary_cross_entropy_with_logits(output, target.float(), reduction="none")

        if self.hparams.use_triplet_loss:
            loss_projection = self.hparams.triplet_alpha * (
                projection_loss(emb_penultimate, target) + self.hparams.triplet_gamma
            )
            loss_regularization = self.hparams.triplet_beta * regularization_loss(emb_penultimate)
            if torch.any(torch.isnan(loss_projection)):
                logger.warning(f"Triplet-loss: projection loss is NaN. (idx={batch_idx}, y_pos_rate={data.y.float().mean().item()})")
                loss_projection = 1
            if torch.any(torch.isnan(loss_regularization)):
                logger.warning(f"Triplet-loss: regularization loss is NaN. (idx={batch_idx}, y_pos_rate={data.y.float().mean().item()})")
                loss_regularization = 1
            loss += loss_projection + loss_regularization

        if not self.hparams.use_smote:
            loss = (batch_class_weights * loss)
        loss = loss.mean()
        return output, target, loss

    def calc_metrics(self, output, target, loss, training=False) -> MetricLog:
        warnings.filterwarnings("error")
        preds = (output > 0.0).long()
        target = target.long()
        pos_rate = self.pos_rate(preds)
        balanced_acc = self.balanced_acc(preds, target)
        prec = -1
        rec = -1
        f1_score = -1
        vd_score = -1
        if not training:
            # only compute these metrics for evaluation
            prec = self.precision(preds, target)
            rec = self.recall(preds, target)
            f1_score = self.f1(preds, target)
            vd_score = self.vd_score(output, target)
        warnings.filterwarnings("default")
        return MetricLog(
            loss=loss,
            prec=prec,
            rec=rec,
            f1_score=f1_score,
            balanced_acc=balanced_acc,
            pos_rate=pos_rate,
            vd_score=vd_score
        )

    def training_step(self, data: Data, batch_idx: int):
        output, target, loss = self.step(data, batch_idx)
        metrics = self.calc_metrics(output, target, loss, training=True)
        metrics.log(self.log, "train", training=True)
        return metrics.loss

    @torch.no_grad()
    def validation_step(self, data: Data, batch_idx: int):
        output, target, loss = self.step(data, batch_idx, eval_metrics=True)
        self.val_scores.append(output)
        self.val_targets.append(target)
        self.val_losses.append(loss)
        return loss

    @torch.no_grad()
    def test_step(self, data: Data, batch_idx: int):
        output, target, loss = self.step(data, batch_idx, eval_metrics=True)
        self.test_scores.append(output)
        self.test_targets.append(target)
        self.test_losses.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        logger.info(
            f"Epoch {self.current_epoch}| "
            f"TRAIN loss={self._trainer.logged_metrics['train/loss']}, "
            f"bal_acc={self._trainer.logged_metrics['train/balanced_accuracy']}, "
            f"pos_rate={self._trainer.logged_metrics['train/positive_rate']} | "
            f"VAL loss={self._trainer.logged_metrics['val/loss']}, "
            f"bal_acc={self._trainer.logged_metrics['val/balanced_accuracy']}, "
            f"f1={self._trainer.logged_metrics['val/f1_score']}, "
            f"prec={self._trainer.logged_metrics['val/precision']}, "
            f"rec={self._trainer.logged_metrics['val/recall']}, "
            f"pos_rate={self._trainer.logged_metrics['val/positive_rate']}"
        )

    def on_validation_epoch_start(self) -> None:
        self.val_scores = []
        self.val_targets = []
        self.val_losses = []

    def on_validation_epoch_end(self) -> None:
        val_scores = torch.cat(self.val_scores)
        val_targets = torch.cat(self.val_targets)
        val_losses = torch.stack(self.val_losses)
        metrics = self.calc_metrics(val_scores, val_targets, loss=val_losses.mean(), training=False)
        log_kwargs = dict(batch_size=len(self.val_losses), on_step=False, on_epoch=True)
        metrics.log(self.log, "val", training=False, log_kwargs=log_kwargs)

    def on_test_epoch_start(self) -> None:
        self.test_scores = []
        self.test_targets = []
        self.test_losses = []

    def on_test_epoch_end(self) -> None:
        test_scores = torch.cat(self.test_scores)
        test_targets = torch.cat(self.test_targets).long()
        test_losses = torch.stack(self.test_losses)
        preds = (test_scores > 0.0).long()
        metrics = self.calc_metrics(test_scores, test_targets, loss=test_losses.mean(), training=False)
        log_kwargs = dict(batch_size=len(self.test_losses), on_step=False, on_epoch=True)
        metrics.log(self.log, "test", training=False, log_kwargs=log_kwargs)
        write_predictions(test_scores, preds, test_targets, self.test_predictions_path)

    def predict_step(self, data: Data, batch_idx: int):
        return self(data)

    def configure_optimizers(self):
        optim = Adam(
            [p for p in self.classifier.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            # TODO: Check if needed
            amsgrad=True,
        )
        return optim

    @staticmethod
    def suggest_optim_hparams(trial: Trial):
        return {
            "lr": trial.suggest_float("hparams.lr", 1e-9, 1e-3, log=True),
            "weight_decay": trial.suggest_float("hparams.weight_decay", 1e-12, 1e-2, log=True),
        }


class NodeTypeEmbedding(nn.Module):
    def __init__(self, vocab_file: Path, one_hot=False):
        super().__init__()
        with open(vocab_file, "r") as f:
            self.vocab = json.load(f)
        self.one_hot = one_hot
        self._dims = len(self.vocab) + 1

    def __call__(self, node_type: str) -> torch.Tensor:
        emb = torch.tensor(self.vocab.get(node_type, len(self.vocab)), dtype=torch.long)
        if self.one_hot:
            emb = F.one_hot(emb, num_classes=self.dims)
        return emb

    @property
    def dims(self):
        return self._dims


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)
