from typing import List, Any
from pathlib import Path
from dataclasses import dataclass

from gnn-baselines.config.models import Word2VecConfig


@dataclass
class Extract:
    in_train: Path
    in_val: Path
    in_test: Path
    out_dir: Path


@dataclass
class Meta:
    raw_dir: Path
    urls_csv: Path
    commits_csv: Path


@dataclass
class Clone:
    urls_csv: Path
    override_urls_csv: Path
    repo_root: Path
    report: Path


@dataclass
class Parse:
    in_dir: Path
    out_dir: Path
    report: Path
    n_jobs: int


@dataclass
class Export:
    in_dir: Path
    out_dir: Path
    report: Path
    n_jobs: int
    joern_bin: Path


@dataclass
class BuildVocab:
    in_dir: Path
    out_file: Path


@dataclass
class Preprocess:
    extract: Extract
    parse: Parse
    export: Export
    build_vocab: BuildVocab
    meta: Meta
    clone: Clone


@dataclass
class TrainW2V:
    in_dir: Path
    w2v_cfg: Word2VecConfig
    ignore_tokens: List[str]
    model_path: Path


@dataclass
class Split:
    mode: str
    dataset_cfg: Any
    val_size: float
    test_size: float
    seed: int
    out_file: Path


@dataclass
class TrainCmd:
    model_name: str
    model_dir: Path
    model_cfg: Any  # can't provide correct type due to circular imports
    dataset_cfg: Any
    experiment_cfg: Any
    hyperparams: Any
    fixed_params: Any
    split_cfg: Any
    train_workers: int
    val_workers: int
    test_workers: int
    test_predictions_path: Path


@dataclass
class Train:
    w2v: TrainW2V
    split: Split
    train: TrainCmd


@dataclass
class CmdConfig:
    preprocess: Preprocess
    train: Train
