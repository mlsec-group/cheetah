import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List

import torch
import torch.nn.functional as F
import transformers
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from graph.datasets.graphs import GraphData
from graph.node_types import DEFAULT_X_TYPES_RAW


@dataclass
class ClassifierConfig:
    hidden_size: int = 256
    num_labels: int = 1
    classifier_dropout: float = 0.1


@dataclass
class GraphModelConfig(PretrainedConfig):
    model_type: str = "graph-model"
    node_encoder_base: str = "microsoft/unixcoder-base-nine"
    max_nodes: int = 378
    clf_config: ClassifierConfig = field(default_factory=lambda: ClassifierConfig())
    x_types: List[str] = field(default_factory=lambda: list(DEFAULT_X_TYPES_RAW))

    # misc hugginface parameters  # TODO remove?
    _attn_implementation_autoset: bool = True
    attn_implementation: str | None = None
    _commit_hash: str | None = None
    transformers_version: str = transformers.__version__

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def to_dict(self):
        config = super().to_dict()
        config["clf_config"] = asdict(self.clf_config)
        return config

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, **kwargs):
        with open(Path(save_directory) / "config.json", "r") as f:
            config = json.load(f)
        config = cls(**config)
        config.clf_config = ClassifierConfig(**config.clf_config)
        return config

    def save_pretrained(self, save_directory: str, **kwargs):
        with open(Path(save_directory) / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)


class GraphModel(PreTrainedModel):
    config_class = GraphModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.node_encoder = AutoModel.from_pretrained(config.node_encoder_base)
        self.freeze_node_encoder()
        self.classifier = self.init_classifier()

    @property
    def graph_encoder(self):
        raise NotImplementedError("graph_encoder must be implemented by subclass")

    @property
    def name(self):
        return self.__class__.__name__

    def freeze_node_encoder(self):
        for param in self.node_encoder.parameters():
            param.requires_grad = False

    def init_classifier(self):
        """Initialize classification head. May be overridden by subclass."""
        return RobertaClassificationHead(self.config.clf_config)

    def forward(
            self,
            x,
            token_idx,
            x_types,
            ast_edge_index,
            cfg_edge_index,
            dfg_edge_index,
            dfg_c_edge_index,
            dfg_r_edge_index,
            dfg_w_edge_index,
            batch,
            return_dict=False, label=None, **kwargs):
        loss = None
        node_repr = self.encode_nodes(x, token_idx)
        graph_repr = self.graph_encoder(
            node_repr,
            x_types,
            ast_edge_index,
            cfg_edge_index,
            dfg_edge_index=dfg_edge_index,
            dfg_c_edge_index=dfg_c_edge_index,
            dfg_r_edge_index=dfg_r_edge_index,
            dfg_w_edge_index=dfg_w_edge_index,
            batch=batch,
        )
        logits = self.classifier(graph_repr.view(graph_repr.size(0), 1, graph_repr.size(1)))
        if label is not None:
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), label.float())
        if return_dict:
            output = SequenceClassifierOutput(logits=logits, loss=loss)
        else:
            output = (loss, logits) if loss is not None else logits
        return output

    @torch.no_grad()
    def encode_nodes(self, x, token_idx):
        tokens, token_mask = batch_node_tokens(x, token_idx)
        embeddings = self.node_encoder(tokens, token_mask).last_hidden_state[:, 0, :]
        return embeddings

    @staticmethod
    def get_collate_fn(device, max_nodes=378, undirected=False):
        def collate_fn(data_list, **kwargs):
            data_list = [to_graph_data(d, max_nodes) for d in data_list]
            data = Batch.from_data_list(data_list)
            if undirected:
                for key in data.keys():
                    if "edge" in key:
                        data[key] = to_undirected(data[key])
            return data.to(device)
        return collate_fn

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, **kwargs):
        save_directory = str(save_directory) if isinstance(save_directory, Path) else save_directory
        config = cls.config_class.from_pretrained(save_directory, **kwargs)
        model = cls(config)
        model.graph_encoder.load_state_dict(torch.load(Path(save_directory) / "graph_encoder.pt", weights_only=True))
        model.classifier.load_state_dict(torch.load(Path(save_directory) / "classifier.pt", weights_only=True))
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.graph_encoder.state_dict(), Path(save_directory) / "graph_encoder.pt")
        torch.save(self.classifier.state_dict(), Path(save_directory) / "classifier.pt")


def to_graph_data(data, max_nodes=1000):
    num_nodes = max(data["token_idx"]) + 1
    node_data = {k: torch.tensor(v, dtype=torch.long) for k, v in data.items() if k in GraphData.node_keys}
    edge_data = {k: _edge_index_to_tensor(v) for k, v in data.items() if k in GraphData.edge_keys}
    label = data["label"]
    data = GraphData(num_nodes=num_nodes, label=label, **node_data, **edge_data)
    data = truncate_graph(data, max_nodes)
    return data


def _edge_index_to_tensor(edge_index):
    """For batching, the edge_index must be of shape (2, num_edges) even if there are no edges."""
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return edge_index


def truncate_graph(data, max_nodes):
    if data.num_nodes > max_nodes:
        tokens_to_keep = torch.where(data.token_idx < max_nodes)[0]
        nodes_to_keep = torch.arange(max_nodes)

        data.x = data.x[tokens_to_keep]
        data.x_types = data.x_types[:max_nodes]
        data.token_idx = data.token_idx[tokens_to_keep]

        for k in GraphData.edge_keys:
            edge_index = data[k]
            keep_idx = torch.isin(edge_index, nodes_to_keep).sum(dim=0) == 2  # only where both nodes are in the truncation
            data[k] = edge_index[:, keep_idx]
        data.num_nodes = max_nodes
    return data


def batch_node_tokens(x, token_idx, truncate=32):
    """Pad tokens to the maximum length in the batch, optionally truncate."""
    token_counts = torch.bincount(token_idx)
    max_tokens = token_counts.max()
    if truncate:
        max_tokens = min(max_tokens, truncate)
    num_nodes = torch.max(token_idx) + 1

    # spread token index: [0, 0, 2, 2, 2,] -> [(0,0), (0,1), (2,0), (2,1), (2,2)]
    token_pos = torch.arange(max_tokens, device=x.device).repeat(num_nodes, 1)
    num_tokens = torch.bincount(token_idx)
    token_mask = torch.arange(max_tokens, device=token_idx.device) < num_tokens[:, None]
    token_pos[~token_mask] = -1
    token_pos = token_pos.view(-1)
    token_pos = token_pos[token_pos != -1]

    node_start_idx = torch.cumsum(num_tokens, dim=0) - num_tokens
    node_end_idx = node_start_idx + torch.minimum(num_tokens, torch.tensor(max_tokens, device=x.device, dtype=x.dtype))

    if truncate:
        token_indices = torch.arange(token_idx.size(0), device=x.device)
        valid_mask = (token_indices >= node_start_idx[token_idx]) & (token_indices < node_end_idx[token_idx])
        truncate_idx = torch.where(valid_mask)[0]
        token_idx = token_idx[truncate_idx]
        x = x[truncate_idx]

    tokens = torch.ones(num_nodes, max_tokens, device=x.device, dtype=x.dtype)
    token_mask = torch.zeros(num_nodes, max_tokens, device=x.device, dtype=x.dtype)
    tokens[token_idx, token_pos] = x
    token_mask[token_idx, token_pos] = 1

    return tokens, token_mask
