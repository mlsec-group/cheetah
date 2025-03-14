from pathlib import Path

import torch

from gnn-baselines.datasets.extractors.base import Extractor


class LabelExtractor(Extractor):
    _keys = ["y"]
    _stat_keys = _keys

    def __init__(self, interim_dir: Path):
        super().__init__(LabelExtractor._keys, interim_dir)
        self.extract_funcs = [self.extract_label]

    @property
    def stat_keys(self):
        return self._stat_keys

    def extract_label(self, in_file, graph, code):
        return {"y": torch.tensor([int(graph["file"].split("_")[0])], dtype=torch.long)}
