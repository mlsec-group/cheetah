import json
from pathlib import Path
from typing import List, Dict

from joblib import Parallel, delayed, parallel_backend

from gnn-baselines.sitter.common import parse, traverse_leaves_dfs


class NCSCorpus:
    """
    Sample tokens from the natural code sequence (NCS) edges
    between AST leaf nodes.
    """

    def __init__(self, processed_dir: Path, ignore: List[str] = None, workers: int = 1):
        self.in_files = list(processed_dir.glob("**/*.json"))
        self.ignore = ignore
        self.workers = workers

    def __len__(self):
        return len(self.in_files)

    def __iter__(self):
        with parallel_backend("loky"):
            yield from Parallel(n_jobs=self.workers)(
                delayed(process_single)(Path(in_file), self.ignore) for in_file in self.in_files
            )

    def get(self, idx):
        return process_single(self.in_files[idx])


def process_single(in_file, ignore):
    with open(in_file, "r") as f:
        graph = json.load(f)[0]
    sentence = []
    for code in traverse_ncs(graph["nodes"]):
        tokens = list(tokenize_str(code, ignore))
        sentence.extend(tokens)
    return sentence


def traverse_ncs(nodes):
    ncs_nodes = {n["ncs"]: n["code"] for n in nodes if "ncs" in n}
    if len(ncs_nodes) == 0:
        return

    max_ncs = max(ncs_nodes.keys())
    for i in range(max_ncs + 1):
        if i in ncs_nodes:
            yield ncs_nodes[i]


class Tokenizer:
    def __init__(self, ignore_token: List[str] = None):
        self.ignore_token = ignore_token

    def __call__(self, string: str):
        return tokenize_str(string, self.ignore_token)


def tokenize_str(s: str, ignore_token: List[str] = None):
    root = parse(s)
    tokens = [n.text.decode() for n in traverse_leaves_dfs(root) if ignore_token is None or n.type not in ignore_token]
    return tokens


def idx_from_tokens(tokens: List[str], vocab: Dict[str, int]):
    return [vocab[tok] for tok in tokens]
