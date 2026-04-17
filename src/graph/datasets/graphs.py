import json
from pathlib import Path
from collections import defaultdict

import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer


class GraphData(Data):
    inc_keys = [
        "token_idx",
        "ast_edge_index",
        "cfg_edge_index",
        "dfg_edge_index",
        "dfg_c_edge_index",
        "dfg_r_edge_index",
        "dfg_w_edge_index",
    ]

    node_keys = [
        "x",
        "x_types",
        "token_idx"
    ]

    edge_keys = [
        "ast_edge_index",
        "cfg_edge_index",
        "dfg_edge_index",
        "dfg_c_edge_index",
        "dfg_r_edge_index",
        "dfg_w_edge_index",
    ]

    def __inc__(self, key, value, *args, **kwargs):
        if key in self.inc_keys:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def get_graph_path(idx, project, split, graph_root):
    return Path(graph_root) / split / project / f"{idx}.json"


def has_graph(sample, graph_root, split):
    idx = sample["idx"]
    project = sample.get("project", "default")
    return get_graph_path(idx, project, split, graph_root).exists()


def get_node_type_vocab(dataset, graph_root):
    node_types = set()
    for split in ["train", "val"]:
        for sample in dataset[split]:
            if not has_graph(sample, graph_root, split):
                continue
            graph_file = get_graph_path(sample["idx"], sample.get("project", "default"), split, graph_root)
            with open(graph_file, "r") as f:
                graph = json.load(f)[0]
            for node in graph["nodes"]:
                label = node.get("label")
                if label is not None:
                    node_types.add(label)
        node_type_vocab = {t: i+1 for i, t in enumerate(sorted(list(node_types)))}
        node_type_vocab[""] = 0
    return node_type_vocab


def to_pyg(samples, graph_root, split, node_type_vocab, base_model, hf_cache, model_max_length):
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=hf_cache, max_length=model_max_length)

    def tokenize(node):
        code = node.get("code", "")
        tokens = tokenizer(code)["input_ids"]
        return torch.tensor(tokens, dtype=torch.long)

    batch_data = {k: [] for k in GraphData.node_keys + GraphData.edge_keys}
    for i in range(len(samples["idx"])):
        graph_file = get_graph_path(samples["idx"][i], samples["project"][i], split, graph_root)
        with open(graph_file, "r") as f:
            graph = json.load(f)[0]

        token_idx = []
        node_id_map = {}
        tokens = []
        node_types = []
        for i, node in enumerate(graph["nodes"]):
            node_tokens = tokenize(node)
            node_id_map[node["id"]] = i
            token_idx.append(torch.ones(len(node_tokens), dtype=torch.long) * i)
            tokens.append(node_tokens)
            node_types.append(node_type_vocab[node.get("label", "UNKNOWN")])

        edges = {}
        add_dfg_x_edges(graph)
        for key in ("ast", "cfg", "dfg", "dfg_c", "dfg_r", "dfg_w"):
            pyg_key = f"{key}_edge_index"
            edges[pyg_key] = []
            if key not in graph:
                continue
            for edge in graph[key]:
                edges[pyg_key].append(torch.tensor([node_id_map[edge["in"]], node_id_map[edge["out"]]], dtype=torch.long).view(2, 1))
            if len(edges[pyg_key]) > 0:
                edges[pyg_key] = torch.cat(edges[pyg_key], dim=1)
            else:
                edges[pyg_key] = torch.zeros(2, 0, dtype=torch.long)

        batch_data["x"].append(torch.cat(tokens))
        batch_data["x_types"].append(torch.tensor(node_types, dtype=torch.long))
        batch_data["token_idx"].append(torch.cat(token_idx, dim=0))
        for key in GraphData.edge_keys:
            batch_data[key].append(edges[key])

    samples.update(batch_data)
    return samples


def add_dfg_x_edges(graph):
    dfg_x_edges = defaultdict(list)
    for node in graph["nodes"]:
        if node.get("dfg_c"):
            dfg_x_edges["dfg_c"].append({"in": node["dfg_c"], "out": node["id"]})
        if node.get("dfg_r"):
            dfg_x_edges["dfg_r"].append({"in": node["dfg_r"], "out": node["id"]})
        if node.get("dfg_w"):
            dfg_x_edges["dfg_w"].append({"in": node["dfg_w"], "out": node["id"]})
    graph.update(dfg_x_edges)
