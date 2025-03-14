from typing import List

import torch
from torch import nn
from torch_geometric.data import Data
from optuna.trial import Trial

from gnn-baselines.models.base import Model


class MCPPEmbed(nn.Module):
    def __init__(self, mcpp_features: List[str], mean, std):
        super().__init__()
        self.mcpp_features = [f"mcpp_{feat}" for feat in mcpp_features]
        self.mean = torch.tensor([mean[key] for key in self.mcpp_features], dtype=torch.float).view(1, -1)
        self.std = torch.tensor([std[key] for key in self.mcpp_features], dtype=torch.float).view(1, -1)

    def forward(self, data: Data):
        vec = torch.stack([data[k] for k in self.mcpp_features], dim=1)
        vec -= self.mean.to(vec)
        vec /= self.std.to(vec)
        return vec


class MCPPHead(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.classifier(z)
        return out


class LinearMCPP(Model):
    _feature_set = "mcpp"

    def __init__(self, name, class_ratio, test_predictions_path, mcpp_features, mean, std, **hparams):
        super().__init__(
            name,
            LinearMCPP._feature_set,
            class_ratio,
            test_predictions_path,
            **hparams
        )
        self.save_hyperparameters({
            "mcpp_features": mcpp_features,
            "mean": mean,
            "std": std
        })
        self.mcpp_embed = MCPPEmbed(mcpp_features, mean, std)
        self.mcpp_head = MCPPHead(len(mcpp_features), self.out_dim)

    @property
    def embedder(self):
        return self.mcpp_embed

    @property
    def classifier(self):
        return self.mcpp_head.classifier

    @staticmethod
    def suggest_hparams(trial: Trial):
        return {}
