from typing import Optional, List
from dataclasses import dataclass

import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops, scatter
from torch.nn import functional as F

from graph.gnn.base import GraphModel, GraphModelConfig


def max_pool_neighbor_x(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    flow: Optional[str] = "source_to_target",
) -> torch.Tensor:
    r"""Max pools neighboring node features, where each feature in
    :obj:`data.x` is replaced by the feature value with the maximum value from
    the central node and its neighbors.
    """
    num_nodes = x.shape[0]
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    row, col = (row, col) if flow == "source_to_target" else (col, row)

    x_new = scatter(x[row], col, dim=0, dim_size=num_nodes, reduce="max")
    return x_new


class DevignHead(nn.Module):
    def __init__(self, graph_dim: int, dropout: float, out_dim: int):
        super().__init__()
        self.graph_dim = graph_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # fmt: off
        self.mlp_z = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.graph_dim, self.out_dim),
        )
        self.mlp_y = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.graph_dim, self.out_dim),
        )
        # fmt: on

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Devign has this weird classification head, see Eq. 7-9 in the original paper.
        """
        Z_out = self.mlp_z(Z[:, 0, :self.graph_dim])
        Y_out = self.mlp_y(Z[:, 0, self.graph_dim:])
        out = torch.mul(Z_out, Y_out)
        out = torch.mean(out, dim=1)
        return out


class DevignConvLayer(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        # fmt: off
        # Graph Convolutional Layersgraphs-debug
        self.ast_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(input_dim, out_dim), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.cfg_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(input_dim, out_dim), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.dfg_c_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(input_dim, out_dim), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.dfg_r_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(input_dim, out_dim), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.dfg_w_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(input_dim, out_dim), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.agg = lambda h_ast, h_cfg, h_dfg_c, h_dfg_r, h_dfg_w: h_ast + h_cfg + h_dfg_c + h_dfg_r + h_dfg_w
        self.gru = nn.GRUCell(out_dim, out_dim)
        # fmt: on

    def forward(
        self,
        h_t: torch.Tensor,
        ast_edge_index: torch.Tensor,
        cfg_edge_index: torch.Tensor,
        dfg_c_edge_index: torch.Tensor,
        dfg_r_edge_index: torch.Tensor,
        dfg_w_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h_ast = self.ast_conv(h_t, ast_edge_index)
        h_cfg = self.cfg_conv(h_t, cfg_edge_index)
        h_dfg_c = self.dfg_c_conv(h_t, dfg_c_edge_index)
        h_dfg_r = self.dfg_r_conv(h_t, dfg_r_edge_index)
        h_dfg_w = self.dfg_w_conv(h_t, dfg_w_edge_index)
        h_t = self.gru(self.agg(h_ast, h_cfg, h_dfg_c, h_dfg_r, h_dfg_w), h_t)
        return h_t


class DevignEncoder(nn.Module):
    def __init__(self, node_dim: int, node_proj_dim: int, graph_dim: int, num_timesteps: int, x_types: List[str]):
        super().__init__()
        self.x_type_vocab = {t: i for i, t in enumerate(x_types)}
        self.pad_dim = graph_dim - node_proj_dim
        assert self.pad_dim >= 0, "graph_dim must be >= node_proj_dim"
        self.num_timesteps = num_timesteps
        # fmt: off
        self.conv_layers = nn.ModuleList(
            [DevignConvLayer(node_proj_dim, graph_dim)] +
            [DevignConvLayer(graph_dim, graph_dim) for _ in range(1, num_timesteps)])

        self.node_proj = nn.Sequential(
            nn.Linear(node_dim + len(self.x_type_vocab), node_proj_dim),
            nn.LeakyReLU(),
        )

        # Pooling of Hidden States
        self.conv_z = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(2*graph_dim, graph_dim), "x, edge_index -> x"),
            nn.LeakyReLU(),
            nn.BatchNorm1d(graph_dim),
            (max_pool_neighbor_x, "x, edge_index -> x"),
            (gnn.GCNConv(graph_dim, graph_dim), "x, edge_index -> x"),
            nn.LeakyReLU(),
            nn.BatchNorm1d(graph_dim),
            (max_pool_neighbor_x, "x, edge_index -> x")
        ])
        self.conv_y = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(graph_dim, graph_dim), "x, edge_index -> x"),
            nn.LeakyReLU(),
            nn.BatchNorm1d(graph_dim),
            (max_pool_neighbor_x, "x, edge_index -> x"),
            (gnn.GCNConv(graph_dim, graph_dim), "x, edge_index -> x"),
            nn.LeakyReLU(),
            nn.BatchNorm1d(graph_dim),
            (max_pool_neighbor_x, "x, edge_index -> x")
        ])
        # fmt: on

    def forward(
        self,
        node_repr,
        node_types,
        ast_edge_index,
        cfg_edge_index,
        dfg_c_edge_index,
        dfg_r_edge_index,
        dfg_w_edge_index,
        batch,
        **kwargs,
    ):
        node_types = F.one_hot(node_types, num_classes=len(self.x_type_vocab)).to(torch.float)
        node_repr = torch.cat([node_repr, node_types], dim=1)
        node_repr = self.node_proj(node_repr)

        h_t = nn.functional.pad(node_repr, (0, self.pad_dim), "constant", value=0)
        for i in range(len(self.conv_layers)):
            h_t = self.conv_layers[i](
                h_t,
                ast_edge_index,
                cfg_edge_index,
                dfg_c_edge_index,
                dfg_r_edge_index,
                dfg_w_edge_index,
            )

        h_T = h_t
        c = torch.cat((h_T, node_repr), dim=-1)
        seq_edges = _sequential_adjancency(batch).to(ast_edge_index)
        Z_l = gnn.global_max_pool(self.conv_z(c, seq_edges), batch=batch)
        Y_l = gnn.global_max_pool(self.conv_y(h_T, seq_edges), batch=batch)

        return torch.concat([Z_l, Y_l], dim=1)


def _sequential_adjancency(batch):
    """
    Small trick to create a 1-D convolution over a sequence of nodes.
    For each graph in the batch, we create sequential connections between consecutive nodes.
    The nodes occur in topological AST order which constitutes a canonical ordering.
    """
    edge_index = []
    offset = 0
    for num_nodes in torch.bincount(batch):
        src = torch.arange(offset, offset + num_nodes - 1, dtype=torch.long).view(1, -1)
        dst = torch.arange(offset + 1, offset + num_nodes, dtype=torch.long).view(1, -1)
        edge_index.append(torch.vstack([src, dst]))
        offset += num_nodes
    edge_index = torch.hstack(edge_index)
    return edge_index


@dataclass
class DevignConfig(GraphModelConfig):
    model_type: str = "devign"
    node_dim: int = 768
    node_proj_dim: int = 256
    graph_dim: int = 256
    num_timesteps: int = 3
    dropout: float = 0.1
    hidden_size: int = 256


class Devign(GraphModel):
    config_class = DevignConfig

    def __init__(self, config: DevignConfig):
        super().__init__(config)
        self._graph_encoder = DevignEncoder(
            config.node_dim,
            config.node_proj_dim,
            config.graph_dim,
            config.num_timesteps,
            config.x_types,
        )

    @property
    def graph_encoder(self):
        return self._graph_encoder

    def init_classifier(self):
        return DevignHead(
            graph_dim=self.config.graph_dim,
            dropout=self.config.dropout,
            out_dim=self.config.hidden_size,
        )
