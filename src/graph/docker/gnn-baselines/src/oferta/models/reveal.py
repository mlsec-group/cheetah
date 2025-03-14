import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from optuna.trial import Trial

from gnn-baselines.models.base import Model, ClassificationHead


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class GlobalAddPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_add_pool(x, batch)


class ExtractFeature(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=out_dim, bias=True), nn.LeakyReLU(), nn.Dropout(dropout)
        )
        self.feature = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features=out_dim, out_features=input_dim, bias=True),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_features=input_dim, out_features=out_dim, bias=True),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out


class RevealEmbed(nn.Module):
    def __init__(self, graph_embed_size, hidden_dim, ggnn_layers, extractor_layers, dropout):
        super().__init__()
        self.ggnn = gnn.GatedGraphConv(out_channels=graph_embed_size, num_layers=ggnn_layers)
        self.readout = GlobalAddPool()
        self.extract_feature = ExtractFeature(graph_embed_size, hidden_dim, extractor_layers, dropout)

    def forward(self, data: Data):
        x = data.x
        edge_index = _merge_edges(data.ast_edge_index, data.cfg_edge_index, data.dfg_edge_index)
        outputs = self.ggnn(x, edge_index)
        pooled = self.readout(outputs, data.batch)
        z = self.extract_feature(pooled)
        return z


class Reveal(Model):
    _feature_set = "devign"

    def __init__(
        self,
        name,
        class_ratio,
        test_predictions_path,
        input_dim,
        hidden_dim,
        ggnn_layers,
        graph_embed_size,
        extractor_layers,
        dropout,
        **hparams,
    ):
        super().__init__(name, Reveal._feature_set, class_ratio, test_predictions_path, **hparams)

        self.save_hyperparameters(
            {
                "hidden_dim": hidden_dim,
                "ggnn_layers": ggnn_layers,
                "extractor_layers": extractor_layers,
                "dropout": dropout,
            }
        )

        self.reveal_embed = RevealEmbed(graph_embed_size, hidden_dim, ggnn_layers, extractor_layers, dropout)
        self.reveal_head = ClassificationHead(hidden_dim, hidden_dim, self.out_dim, dropout).classifier

    @property
    def embedder(self):
        return self.reveal_embed

    @property
    def classifier(self):
        return self.reveal_head

    @staticmethod
    def suggest_hparams(trial: Trial):
        hparams = {
            "hidden_dim": trial.suggest_int("models.reveal.hidden_dim", 64, 512),
            "ggnn_layers": trial.suggest_int("models.reveal.ggnn_layers", 1, 6),
            "graph_embed_size": trial.suggest_int("models.reveal.graph_embed_size", 200, 400),
            "extractor_layers": trial.suggest_int("models.reveal.extractor_layers", 1, 3),
            "dropout": trial.suggest_float("models.reveal.dropout", 0.0, 1.0),
        }
        return hparams


def _merge_edges(ast_edge_index, cfg_edge_index, dfg_edge_index):
    """Combine different edge types and remove duplicate edges."""
    merged = torch.hstack([ast_edge_index, cfg_edge_index, dfg_edge_index])
    return torch.unique(merged, dim=1)
