"""
Schema definition and validation of the hierarchical config files.
"""
from typing import List
from pathlib import Path
from dataclasses import dataclass

from gnn_baselines.config.cmd import CmdConfig
from gnn_baselines.config.models import ModelConfig


@dataclass
class Experiment:
    dataset_name: str
    model_name: str
    seed: int
    use_smote: bool
    use_triplet_loss: bool
    test_predictions_path: Path


@dataclass
class Hardware:
    cores: int


@dataclass
class HyperParameters:
    lr: float
    accumulated_batches: int
    batch_size: int
    weight_decay: float
    triplet_alpha: float
    triplet_beta: float
    triplet_gamma: float


@dataclass
class FixedParameters:
    max_epochs: int
    checkpoint_interval: int
    patience: int
    val_size: float
    test_size: float
    embed_model: Path


@dataclass
class TokenizerConfig:
    ignore_tokens: List[str]


@dataclass
class Paths:
    data_root: Path
    raw: Path
    meta: Path
    interim: Path
    cpg: Path
    processed: Path
    models: Path
    logs: Path
    reports: Path
    llm_cache: Path
    repo_root: Path


@dataclass
class Dataset:
    name: str
    raw_train: Path
    raw_val: Path
    raw_test: Path
    root: Path
    processed_file: Path | None
    node_type_vocab: Path
    w2v_kv_file: Path
    tokenizer_cfg: TokenizerConfig
    interim_dir: Path
    jobs: int


@dataclass
class Config:
    paths: Paths
    hparams: HyperParameters
    fixed_params: FixedParameters
    hardware: Hardware
    experiment: Experiment
    tokenizer: TokenizerConfig
    dataset: Dataset
    models: ModelConfig
    cmd: CmdConfig
