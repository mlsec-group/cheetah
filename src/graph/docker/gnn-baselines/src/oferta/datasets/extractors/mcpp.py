from pathlib import Path

import torch
from mcpp import extract_code

from gnn-baselines.datasets.extractors.base import Extractor


class McppExtractor(Extractor):
    _keys = [
        "mcpp_C1",
        "mcpp_C2",
        "mcpp_C3",
        "mcpp_C4",
        "mcpp_V1",
        "mcpp_V2",
        "mcpp_V3",
        "mcpp_V4",
        "mcpp_V5",
        "mcpp_V6",
        "mcpp_V7",
        "mcpp_V8",
        "mcpp_V9",
        "mcpp_V10",
        "mcpp_V11"
    ]
    _stat_keys = _keys

    def __init__(self, interim_dir: Path):
        super().__init__(McppExtractor._keys, interim_dir)
        self.extract_funcs = [self.extract_mcpp]

    @property
    def stat_keys(self):
        return self._stat_keys

    def extract_mcpp(self, in_file, graph, code):
        mcpp_features = extract_code(code)
        features = {f"mcpp_{key}": torch.tensor([val], dtype=torch.float32) for key, val in mcpp_features.items()}
        return features
