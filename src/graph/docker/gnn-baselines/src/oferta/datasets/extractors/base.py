from abc import ABC, abstractmethod

import json
from pathlib import Path


class Extractor(ABC):
    def __init__(self, keys, interim_dir):
        self.keys = keys
        self.interim_dir = interim_dir

    def __call__(self, in_file, graph, code):
        feat = {}
        for extract_func in self.extract_funcs:
            feat.update(extract_func(in_file, graph, code))
        return feat

    def read_data(self, in_file: Path):
        return _read_data(in_file, self.interim_dir)

    @property
    @abstractmethod
    def stat_keys(self):
        pass


def get_orig_code(interim_dir: Path, file_name: str):
    orig_file = list(interim_dir.glob(f"**/{file_name}"))[0]  # TODO what if original file is not found?
    with open(orig_file, "r") as f:
        code = f.read()
    return code


def _read_data(in_file: Path, interim_dir: Path):
    with open(in_file, "r") as f:
        graph = json.load(f)[0]
    code = get_orig_code(interim_dir, graph['file'])
    return graph, code
