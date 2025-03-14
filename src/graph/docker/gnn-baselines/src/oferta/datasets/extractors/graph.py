from pathlib import Path

import torch
import logging
from loguru import logger as log
from torch_geometric.utils._scatter import scatter
from torch_geometric.data import Data
import gensim.models.keyedvectors as gensim_models
import numpy as np

from gnn-baselines.config.schema import TokenizerConfig
from gnn-baselines.train.tokenize import Tokenizer
from gnn-baselines.datasets.extractors.base import Extractor


# disable gensim logs
logging.getLogger(gensim_models.__name__).setLevel(logging.ERROR)


class GraphExtractor(Extractor):
    _keys = [
        "x",
        "x_types",
        "token_idx",
        "subtokens",
        "ast_edge_index",
        "cfg_edge_index",
        "dfg_edge_index",
        "dfg_c_edge_index",
        "dfg_r_edge_index",
        "dfg_w_edge_index",
        "name",
        "joern_ids",
        "line_numbers"
    ]

    _stat_keys = [
        "x",
        "x_types"
    ]

    def __init__(self, interim_dir: Path, tokenizer_cfg: TokenizerConfig, vocab, node_type_embedding, w2v_kv_file):
        super().__init__(GraphExtractor._keys, interim_dir)
        self.tokenizer_cfg = tokenizer_cfg
        self.vocab = vocab
        self.node_type_embedding = node_type_embedding
        self.extract_funcs = [self.extract_graph]
        self.embed = SubTokenEmbedding(w2v_kv_file, aggregation="mean")

    @property
    def stat_keys(self):
        return self._stat_keys

    def extract_graph(self, in_file, graph, code):
        x, token_idx, subtokens, id_map = get_subtokens(graph["nodes"], self.tokenizer_cfg, self.vocab)
        x_types = torch.vstack([self.node_type_embedding(node["label"]) for node in graph["nodes"]])
        ast_edge_index = get_edge_index(graph["ast"], x.shape[0], id_map)
        cfg_edge_index = get_edge_index(graph["cfg"], x.shape[0], id_map)
        dfg_edge_index = get_edge_index(graph["dfg"], x.shape[0], id_map)
        joern_ids, line_numbers = get_extra_info(graph["nodes"])
        extra_edges = get_extra_edges(graph["nodes"], id_map)

        features = dict(
            x=x,
            x_types=x_types,
            token_idx=token_idx,
            subtokens=subtokens,
            ast_edge_index=ast_edge_index,
            cfg_edge_index=cfg_edge_index,
            dfg_edge_index=dfg_edge_index,
            dfg_c_edge_index=extra_edges["dfg_c"],
            dfg_r_edge_index=extra_edges["dfg_r"],
            dfg_w_edge_index=extra_edges["dfg_w"],
            name=in_file.name,
            joern_ids=joern_ids,
            line_numbers=line_numbers,
        )
        # replaces x (node IDs) by node embedding vectors
        features["x"] = self.embed.embed_x(features["x"], features["token_idx"], features["subtokens"])
        return features


def get_subtokens(nodes, tokenizer_cfg: TokenizerConfig, vocab):
    tokenizer = Tokenizer(tokenizer_cfg.ignore_tokens)

    token_idx = []
    subtokens = []
    id_map = {}
    OOV = len(vocab)
    for i, node in enumerate(nodes):
        code = node["code"]
        _subtokens = tokenizer(code)
        _subtokens = [vocab.get(tok, OOV) for tok in _subtokens]
        if len(_subtokens) == 0:
            # add at least a virtual OOV token if all subtokens have been filtered out
            _subtokens = [OOV]
        subtokens.extend(_subtokens)
        token_idx.extend([i] * len(_subtokens))
        id_map[node["id"]] = i

    x = torch.tensor(list(range(len(nodes))), dtype=torch.long)
    token_idx = torch.tensor(token_idx, dtype=torch.long)
    subtokens = torch.tensor(subtokens, dtype=torch.long)
    return x, token_idx, subtokens, id_map


def get_edge_index(edges, num_nodes, id_map):
    edge_index_list = []
    for edge in edges:
        try:
            src = id_map[edge["in"]]
            dst = id_map[edge["out"]]
        except KeyError:
            continue
        if src < num_nodes and dst < num_nodes:
            edge_index_list.append([src, dst])
        else:
            log.warning(f"edge ({src}, {dst}) can't be in graph with {num_nodes} nodes.")
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
    return edge_index


def get_extra_info(nodes):
    joern_ids = []
    line_numbers = []
    for node in nodes:
        joern_ids.append(node["id"])
        line_numbers.append(node.get("lineNumber", -1))
    joern_ids = torch.tensor(joern_ids, dtype=torch.long)
    line_numbers = torch.tensor(line_numbers, dtype=torch.long)
    return joern_ids, line_numbers


def get_extra_edges(nodes, id_map):
    """Get all the extra edges from the devign paper."""
    dfg_c = []
    dfg_r = []
    dfg_w = []
    for node in nodes:
        n_id = node["id"]
        if "dfg_c" in node:
            dfg_c.append([id_map[node["dfg_c"]], id_map[n_id]])
        if "dfg_r" in node:
            dfg_r.append([id_map[node["dfg_r"]], id_map[n_id]])
        if "dfg_w" in node:
            dfg_w.append([id_map[node["dfg_w"]], id_map[n_id]])
    dfg_c = torch.tensor(dfg_c, dtype=torch.long).t()
    dfg_r = torch.tensor(dfg_r, dtype=torch.long).t()
    dfg_w = torch.tensor(dfg_w, dtype=torch.long).t()
    return {"dfg_c": dfg_c, "dfg_r": dfg_r, "dfg_w": dfg_w}


class SubTokenEmbedding(object):
    """Embedding torch transformation."""

    def __init__(self, w2v_kv_file: Path, aggregation: str = "mean"):
        self.kv = gensim_models.KeyedVectors.load(str(w2v_kv_file))
        some_key = self.kv.index_to_key[0]
        self.dims = self.kv[some_key].shape[-1]
        self.aggregation = aggregation
        self.oov_vector = np.zeros((self.dims,), dtype=float)

    def __call__(self, data: Data):
        """ callable allows to be used as torch transform """
        data.x = self.embed_x(data.x, data.token_idx, data.subtokens)
        return data

    def embed_x(self, x, token_idx, subtokens):
        emb_subtokens = np.vstack(
            [self.kv[tok.item()] if tok.item() in self.kv else self.oov_vector for tok in subtokens]
        )
        x = scatter(torch.tensor(emb_subtokens, dtype=torch.float32), token_idx, reduce=self.aggregation)
        # TODO: hotfix
        num_empty = x.shape[0] - (token_idx.max() + 1)
        if num_empty > 0:
            x = torch.cat([x, torch.zeros(num_empty, x.shape[1], dtype=x.dtype)], dim=0)
        return x