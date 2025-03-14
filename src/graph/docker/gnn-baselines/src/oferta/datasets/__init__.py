from gnn-baselines.datasets.primevul import PrimeVul
from gnn-baselines.datasets.base import DataModule
from gnn-baselines.config.cmd import Split

DATASETS = {
    "primevul": PrimeVul,
}


def load_dataset(
    name: str, split_cfg: Split, batch_size, train_workers, val_workers, test_workers, seed, **kwargs
) -> DataModule:
    if name not in DATASETS:
        raise ValueError(f"{name} not in datasets. Must be one of {sorted(DATASETS.keys())}")

    ds = DATASETS[name](**kwargs)
    data_module = DataModule(
        ds,
        split_cfg,
        batch_size,
        train_workers,
        val_workers,
        test_workers,
        seed=seed,
    )
    return data_module
