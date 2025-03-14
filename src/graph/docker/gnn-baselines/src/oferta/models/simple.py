from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from optuna.trial import Trial

from gnn-baselines.models.base import Model, ClassificationHead


class SimpleEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data):
        return gnn.pool.global_mean_pool(data.x, batch=data.batch)


class SimpleModel(Model):
    _feature_set = "devign"

    def __init__(self, name, class_ratio, test_predictions_path, input_dim, hidden_dim, dropout, **hparams):
        super().__init__(name, SimpleModel._feature_set, class_ratio, test_predictions_path, **hparams)
        self.save_hyperparameters({
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout
        })

        self.simple_embed = SimpleEmbed()
        self.simple_head = ClassificationHead(input_dim, hidden_dim, self.out_dim, dropout)

    @property
    def embedder(self):
        return self.simple_embed

    @property
    def classifier(self):
        return self.simple_head.classifier

    @staticmethod
    def suggest_hparams(trial: Trial):
        hparams = {
            "hidden_dim": trial.suggest_int("models.simple.hidden_dim", 64, 512),
            "dropout": trial.suggest_float("models.simple.dropout", 0.0, 1.0),
        }
        return hparams
