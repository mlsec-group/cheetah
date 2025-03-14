from typing import Optional

import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops, scatter
from torch_geometric.data import Data
from optuna.trial import Trial

from gnn-baselines.models.base import Model


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
    def __init__(self, graph_embed_size: int, concat_dim: int, dropout: float,out_dim: int):
        super().__init__()
        self.graph_embed_size = graph_embed_size
        self.concat_dim = concat_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # fmt: off
        self.mlp_z = nn.Sequential(
            nn.Linear(self.concat_dim, self.concat_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.concat_dim, self.concat_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.concat_dim, self.out_dim),
        )
        self.mlp_y = nn.Sequential(
            nn.Linear(self.graph_embed_size, self.graph_embed_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.graph_embed_size, self.graph_embed_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.graph_embed_size, self.out_dim),
        )
        # fmt: on

    def _forward_penultimate(self, Z: torch.Tensor) -> torch.Tensor:
        Z_l = Z[:, : self.concat_dim]
        Y_l = Z[:, self.concat_dim:]
        Z_penu = self.mlp_z[:-1](Z_l)
        Y_penu = self.mlp_y[:-1](Y_l)

        return torch.cat([Z_penu, Y_penu], dim=1)

    def _forward_ultimate(self, Z: torch.Tensor) -> torch.Tensor:
        Z_penu = Z[:, : self.concat_dim]
        Y_penu = Z[:, self.concat_dim:]
        Z_out = self.mlp_z[-1](Z_penu)
        Y_out = self.mlp_y[-1](Y_penu)

        out = torch.mul(Z_out, Y_out)
        out = torch.mean(out, dim=1)
        return out

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        Z_penu = self._forward_penultimate(Z)
        out = self._forward_ultimate(Z_penu)
        return out

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # only -1 implemented for penultimate layer
            if idx.stop == -1:
                return self._forward_penultimate
        elif isinstance(idx, int):
            if idx == -1:
                return self._forward_ultimate


class DevignConvLayer(nn.Module):
    def __init__(self, graph_embed_size: int):
        super().__init__()
        # fmt: off
        # Graph Convolutional Layers
        self.ast_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(graph_embed_size, graph_embed_size), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.cfg_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(graph_embed_size, graph_embed_size), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.dfg_c_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(graph_embed_size, graph_embed_size), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.dfg_r_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(graph_embed_size, graph_embed_size), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.dfg_w_conv = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(graph_embed_size, graph_embed_size), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
        ])
        self.agg = lambda h_ast, h_cfg, h_dfg_c, h_dfg_r, h_dfg_w: h_ast + h_cfg + h_dfg_c + h_dfg_r + h_dfg_w
        self.gru = nn.GRUCell(graph_embed_size, graph_embed_size)
        # fmt: on

    def forward(self, h_t: torch.Tensor, data: Data) -> torch.Tensor:
        h_ast = self.ast_conv(h_t, data.ast_edge_index)
        h_cfg = self.cfg_conv(h_t, data.cfg_edge_index)
        h_dfg_c = self.dfg_c_conv(h_t, data.dfg_c_edge_index)
        h_dfg_r = self.dfg_r_conv(h_t, data.dfg_r_edge_index)
        h_dfg_w = self.dfg_w_conv(h_t, data.dfg_w_edge_index)
        h_t = self.gru(self.agg(h_ast, h_cfg, h_dfg_c, h_dfg_r, h_dfg_w), h_t)
        return h_t


class DevignEmbed(nn.Module):
    def __init__(self, input_dim: int, graph_embed_size: int, concat_dim: int, num_timesteps: int):
        super().__init__()
        self.pad_dim = graph_embed_size - input_dim
        self.num_timesteps = num_timesteps
        # fmt: off
        # Graph Convolutional Layers
        self.conv_layers = nn.ModuleList([DevignConvLayer(graph_embed_size) for _ in range(num_timesteps)])

        # Pooling of Hidden States
        self.conv_z = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(concat_dim, concat_dim), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(concat_dim),
            (max_pool_neighbor_x, "x, edge_index -> x"),
            (gnn.GCNConv(concat_dim, concat_dim), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(concat_dim),
            (max_pool_neighbor_x, "x, edge_index -> x")
        ])
        self.conv_y = gnn.Sequential("x, edge_index", [
            (gnn.GCNConv(graph_embed_size, graph_embed_size), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(graph_embed_size),
            (max_pool_neighbor_x, "x, edge_index -> x"),
            (gnn.GCNConv(graph_embed_size, graph_embed_size), "x, edge_index -> x"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(graph_embed_size),
            (max_pool_neighbor_x, "x, edge_index -> x")
        ])
        # fmt: on

    def forward(self, data: Data):
        x = torch.concat([data.x, data.x_types], dim=1)
        batch = data.batch

        h_t = nn.functional.pad(x, (0, self.pad_dim), "constant", value=0)
        for i in range(len(self.conv_layers)):
            h_t = self.conv_layers[i](h_t, data)

        h_T = h_t
        c = torch.cat((h_T, x), dim=-1)
        seq_edges = _sequential_adjancency(batch).to(data.ast_edge_index)
        Z_l = gnn.global_max_pool(self.conv_z(c, seq_edges), batch=batch)
        Y_l = gnn.global_max_pool(self.conv_y(h_T, seq_edges), batch=batch)

        return torch.concat([Z_l, Y_l], dim=1)


def _sequential_adjancency(batch):
    edge_index = []
    offset = 0
    for num_nodes in torch.bincount(batch):
        src = torch.arange(offset, offset + num_nodes - 1, dtype=torch.long).view(1, -1)
        dst = torch.arange(offset + 1, offset + num_nodes, dtype=torch.long).view(1, -1)
        edge_index.append(torch.vstack([src, dst]))
        offset += num_nodes
    edge_index = torch.hstack(edge_index)
    return edge_index


class Devign(Model):
    _feature_set = "devign"

    def __init__(
        self,
        name,
        class_ratio,
        input_dim,
        type_dim,
        graph_embed_size,
        num_timesteps,
        test_predictions_path,
        **hparams,
    ):
        super().__init__(name, Devign._feature_set, class_ratio, test_predictions_path, **hparams)
        input_dim = input_dim + type_dim
        concat_dim = input_dim + graph_embed_size

        self.save_hyperparameters(
            {
                "input_dim": input_dim,
                "graph_embed_size": graph_embed_size,
                "concat_dim": concat_dim,
                "out_dim": self.out_dim,
                "num_timesteps": num_timesteps,
            }
        )

        self.devign_embed = DevignEmbed(input_dim, graph_embed_size, concat_dim, num_timesteps)
        self.devign_head = DevignHead(graph_embed_size, concat_dim, self.hparams.dropout, self.out_dim)

    @property
    def embedder(self):
        return self.devign_embed

    @property
    def classifier(self):
        return self.devign_head

    @staticmethod
    def suggest_hparams(trial: Trial):
        hparams = {
            "num_timesteps": trial.suggest_int("models.devign.num_timesteps", 1, 6),
            "graph_embed_size": trial.suggest_int("models.devign.graph_embed_size", 32, 512),
            "dropout": trial.suggest_float("models.devign.dropout", 0.0, 0.5),
        }
        return hparams
