import os
from abc import ABC

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from optuna import Trial

from gnn-baselines.config.cmd import Split


class Dataset(ABC, InMemoryDataset):
    def __init__(self, name, num_classes, feature_set, *args, seed=None, **kwargs):
        self._name = name
        self._num_classes = num_classes
        self._feature_set = feature_set
        super().__init__(*args, **kwargs)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def name(self):
        return self._name

    @property
    def feature_set(self):
        return self._feature_set

    @property
    def data_files(self):
        raise NotImplementedError()


class DataModule(LightningDataModule):
    SUBSAMPLE_TRAIN = 0.9

    def __init__(
        self,
        dataset: Dataset,
        split_cfg: Split,
        batch_size: int,
        train_workers: int,
        val_workers: int,
        test_workers: int,
        seed: int
    ):
        super().__init__()
        self.dataset = dataset
        self.seed = seed
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers
        self.save_hyperparameters({
            "batch_size": batch_size,
            "split_mode": split_cfg.mode,
            "val_size": split_cfg.val_size,
            "test_size": split_cfg.test_size,
            "split_seed": split_cfg.seed
        })
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_idx, self.val_idx, self.test_idx = self.split()

    def split(self):
        data_files = self.dataset.data_files
        assert self.hparams.val_size + self.hparams.test_size < 1
        split_funcs = {"random": get_random_split, "project": get_project_split, "predefined": get_predefined_split}
        if self.hparams.split_mode not in split_funcs:
            raise ValueError(f"mode must be in {list(split_funcs.keys())} but was {self.hparams.split_mode}.")

        train_idx, val_idx, test_idx = split_funcs[self.hparams.split_mode](
            data_files, self.hparams.val_size, self.hparams.test_size, self.hparams.split_seed
        )
        return train_idx, val_idx, test_idx

    def train_dataloader(self):
        idx = self.train_idx
        if self.SUBSAMPLE_TRAIN < 1:
            np.random.seed(self.seed)
            idx = np.random.choice(idx, size=int(len(idx) * self.SUBSAMPLE_TRAIN), replace=False)
        train_ds = self.dataset[idx]
        return DataLoader(
            train_ds, batch_size=self.hparams.batch_size, num_workers=self.train_workers, pin_memory=False, shuffle=True
        )

    def val_dataloader(self):
        val_ds = self.dataset[self.val_idx]
        return DataLoader(val_ds, batch_size=self.hparams.batch_size, num_workers=self.val_workers, pin_memory=False)

    def test_dataloader(self):
        test_ds = self.dataset[self.test_idx]
        return DataLoader(test_ds, batch_size=self.hparams.batch_size, num_workers=self.test_workers, pin_memory=False)

    @staticmethod
    def suggest_data_hparams(trial: Trial):
        return {
            "batch_size": trial.suggest_int("hparams.batch_size", 32, 256, log=True)
        }


def get_random_split(data_files, val_size, test_size, seed=42):
    # initial split into train+val and test
    indices = list(range(len(data_files)))
    trainval_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)

    # find correct val ratio relative to the total number of input files
    num_val = len(indices) * val_size
    val_size = num_val / len(trainval_idx)
    assert 0 < val_size and val_size < 1

    # split train and val
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_size, random_state=seed)

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    # no overlap allowed
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    return train_idx, val_idx, test_idx


def get_project_split(data_files, val_size, test_size, seed=42):
    indices = np.array(list(range(len(data_files))))
    labels = np.array([int(f.stem.split("_")[0]) for f in data_files])
    projects = [f.parent.parent.stem for f in data_files]
    enc = LabelEncoder()
    project_ids = enc.fit_transform(projects)

    # initial split into train+val and test
    traintest_gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = list(traintest_gss.split(indices, labels, project_ids))[0]

    # find correct val ratio relative to the total number of input files
    num_val = len(data_files) * val_size
    val_size = num_val / len(trainval_idx)
    assert 0 < val_size and val_size < 1

    # split train and val
    trainval_project_ids = project_ids[trainval_idx]
    trainval_labels = labels[trainval_idx]
    trainval_gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    _train_idx, _val_idx = list(trainval_gss.split(trainval_idx, trainval_labels, trainval_project_ids))[0]
    train_idx = indices[trainval_idx][_train_idx]
    val_idx = indices[trainval_idx][_val_idx]

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    train_files = [data_files[i] for i in train_idx]
    val_files = [data_files[i] for i in val_idx]
    test_files = [data_files[i] for i in test_idx]

    # no overlap allowed
    assert len(set(train_files) & set(test_files)) == 0
    assert len(set(train_files) & set(val_files)) == 0
    assert len(set(val_files) & set(test_files)) == 0

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    return train_idx, val_idx, test_idx


def get_predefined_split(data_files, val_size=None, test_size=None, seed=None):
    train_idx = []
    val_idx = []
    test_idx = []
    for i, data_file in enumerate(data_files):
        # paths of type **/SPLIT/PROJECT/NAME.c
        split_folder = data_file.parents[2]
        # only keep the part after last slash
        split_folder = os.path.split(split_folder)[1]
        if split_folder == "train":
            train_idx.append(i)
        elif split_folder == "val":
            val_idx.append(i)
        elif split_folder == "test":
            test_idx.append(i)
        else:
            raise RuntimeError(f"Can't assign split to file {data_file} with parent {split_folder}")
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    return train_idx, val_idx, test_idx
