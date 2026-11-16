from dataclasses import dataclass
from typing import List

import torch
import torch_geometric.nn as gnn
from torch import nn
from torch.nn import functional as F

from graph.gnn.base import GraphModel, GraphModelConfig


class DummyEncoder(nn.Module):
    """
    Very simple encoder that performs:
    - two steps of graph convolution on the AST and CFG separately
    - global mean pooling of the concatenated AST and CFG embeddings
    - linear projection to the graph dimension
    """
    def __init__(self, x_types: List[str], node_dim: int, graph_dim: int):
        super().__init__()
        self.x_type_vocab = {t: i for i, t in enumerate(x_types)}
        input_dim = node_dim + len(self.x_type_vocab)
        # fmt: off
        self.ast_gcn = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(input_dim, node_dim), "x, edge_index -> x"),
            nn.GELU(),
            (gnn.GCNConv(node_dim, node_dim), "x, edge_index -> x"),
            nn.GELU(),
        ])
        self.cfg_gcn = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(input_dim, node_dim), "x, edge_index -> x"),
            nn.GELU(),
            (gnn.GCNConv(node_dim, node_dim), "x, edge_index -> x"),
            nn.GELU(),
        ])

        self.out_projection = nn.Linear(node_dim*2, graph_dim)
        # fmt: on

    def forward(self, node_repr, node_types, ast_edge_index, cfg_edge_index, batch, **kwargs):
        node_types = F.one_hot(node_types, num_classes=len(self.x_type_vocab)).to(torch.float)
        node_repr = torch.cat([node_repr, node_types], dim=1)
        h_ast = self.ast_gcn(node_repr, ast_edge_index)
        h_cfg = self.cfg_gcn(node_repr, cfg_edge_index)
        h_graph = gnn.global_mean_pool(torch.cat([h_ast, h_cfg], dim=1), batch)
        h_graph = self.out_projection(h_graph)
        return h_graph


@dataclass
class DummyModelConfig(GraphModelConfig):
    model_type: str = "dummy"
    node_dim: int = 768  # input dim for node representations
    graph_dim: int = 256  # embedding dimension of the whole graph, must equal to hidden_size in classifier


class DummyModel(GraphModel):
    config_class = DummyModelConfig

    def __init__(self, config: DummyModelConfig):
        super().__init__(config)
        self._graph_encoder = DummyEncoder(
            config.x_types,
            config.node_dim,
            config.graph_dim,
        )

    @property
    def graph_encoder(self):
        return self._graph_encoder
