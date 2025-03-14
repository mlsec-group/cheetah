import json
from pathlib import Path
from typing import Set


def get_node_labels(graph) -> Set[str]:
    types = set()
    for node in graph["nodes"]:
        types.add(node["label"])
    return types


def build_vocab(in_dir: Path, out_file: Path):
    in_files = list(in_dir.glob("**/*.json"))
    # exclude test files
    in_files = [f for f in in_files if not any(part == "test" for part in f.parts)]
    all_node_labels = set()
    for in_file in in_files:
        with open(in_file, "r") as f:
            graph = json.load(f)[0]
        all_node_labels |= get_node_labels(graph)

    vocab = {t: i for i, t in enumerate(sorted(all_node_labels))}
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        json.dump(vocab, f, indent=4)
