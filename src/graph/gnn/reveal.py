from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import scatter

from graph.gnn.base import GraphModel, GraphModelConfig


class GlobalPool(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, batch, size):
        return scatter(x, batch, dim=-2, dim_size=size, reduce=self.reduction)


class ExtractFeature(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.feature = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features=out_dim, out_features=out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_features=out_dim, out_features=out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

    @torch.compile(fullgraph=True)
    def forward(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out


class RevealEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        node_proj_dim,
        graph_embed_size,
        hidden_size,
        ggnn_layers,
        extractor_layers,
        dropout,
        x_types,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.graph_embed_size = graph_embed_size
        self.x_type_vocab = {t: i for i, t in enumerate(x_types)}
        self.node_proj = nn.Linear(input_dim + len(self.x_type_vocab), node_proj_dim)
        self.ggnn = gnn.GatedGraphConv(out_channels=graph_embed_size, num_layers=ggnn_layers, aggr="mean")
        self.readout = GlobalPool(reduction="sum")
        self.extract_feature = ExtractFeature(graph_embed_size, hidden_size, extractor_layers, dropout)

    def forward(self, node_repr, node_types, ast_edge_index, cfg_edge_index, dfg_edge_index, batch, **kwargs):
        node_types = F.one_hot(node_types, num_classes=len(self.x_type_vocab)).to(torch.float)
        node_repr = torch.cat([node_repr, node_types], dim=1)
        node_repr = self.node_proj(node_repr)
        edge_index = merge_edges(ast_edge_index, cfg_edge_index, dfg_edge_index)
        node_repr = self.ggnn(node_repr, edge_index)
        graph_repr = self.readout(node_repr, batch, size=batch.max() + 1)
        graph_repr = self.extract_feature(graph_repr)
        return graph_repr


def merge_edges(ast_edge_index: torch.Tensor, cfg_edge_index: torch.Tensor, dfg_edge_index: torch.Tensor) -> torch.Tensor:
    edges = torch.hstack([ast_edge_index, cfg_edge_index, dfg_edge_index])
    return torch.unique(edges, dim=1)


@dataclass
class RevealConfig(GraphModelConfig):
    model_type: str = "reveal"
    node_dim: int = 768
    node_proj_dim: int = 256
    graph_embed_size: int = 256  # must be at least node_proj_dim
    ggnn_layers: int = 3
    extractor_layers: int = 2
    dropout: float = 0
    hidden_size: int = 256


class Reveal(GraphModel):
    config_class = RevealConfig

    def __init__(self, config: RevealConfig):
        super().__init__(config)
        self._graph_encoder = RevealEncoder(
            config.node_dim,
            config.node_proj_dim,
            config.graph_embed_size,
            config.hidden_size,
            config.ggnn_layers,
            config.extractor_layers,
            config.dropout,
            config.x_types,
        )

    @property
    def graph_encoder(self):
        return self._graph_encoder
