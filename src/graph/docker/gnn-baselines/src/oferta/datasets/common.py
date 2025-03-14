from pathlib import Path

import torch
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from gnn-baselines.datasets.base import Dataset
from gnn-baselines.datasets.feature_set import FeatureSet


def load_splits(split_file: Path):
    train_idx, val_idx, test_idx = torch.load(split_file)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def create_dataloader(dataset: Dataset, split: torch.Tensor, batch_size=1, workers=0):
    sampler = RandomSampler(dataset[split])
    return DataLoader(dataset[split], batch_size=batch_size, sampler=sampler, num_workers=workers, shuffle=True)


def data_from_json(in_file, feature_set, **feature_set_kwargs):
    feature_set = FeatureSet.by_name(feature_set, **feature_set_kwargs)
    data = {}
    # read data from HDD only once for all extractors
    graph, code = feature_set.extractors[0].read_data(in_file)
    for extractor in feature_set.extractors:
        data.update(extractor(in_file, graph, code))
    return Data(**data)
