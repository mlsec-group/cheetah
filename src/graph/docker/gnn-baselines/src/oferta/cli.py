"""
Main Command-Line-Interface using Hydra
"""

import sys
from typing import List
from pathlib import Path

import typer
import hydra
from loguru import logger as log
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from gnn-baselines.config.schema import Config
from gnn-baselines.preprocess.extract import extract
from gnn-baselines.preprocess.parse import parse_cpgs
from gnn-baselines.preprocess.export import export_cpgs
from gnn-baselines.preprocess.build_vocab import build_vocab
from gnn-baselines.train.w2v import train_w2v
from gnn-baselines.train.trainer import train as run_train
from gnn-baselines.datasets.primevul import PrimeVul

cfg = None
app = typer.Typer(pretty_exceptions_show_locals=False)  # TODO re-enable?


"""
    Preprocessing
"""

preprocess = typer.Typer()
app.add_typer(preprocess, name="preprocess", help="data preprocessing steps")


@preprocess.command(name="extract")
def extract_cli(overrides: List[str] = typer.Argument(default=None, help="override values for hydra")):
    """
    Extract functions from raw input file.
    """
    cfg = get_cfg(to_remove=2)
    OmegaConf.to_object(cfg)
    log.info(f"Starting raw extraction using\n{_cfg_to_str(cfg.cmd.preprocess.extract)}")
    extract(**cfg.cmd.preprocess.extract)


@preprocess.command(name="parse")
def parse_cli(overrides: List[str] = typer.Argument(default=None, help="override values for hydra")):
    """
    Parse CPGs from input source files.
    """
    cfg = get_cfg(to_remove=2)
    OmegaConf.to_object(cfg)
    log.info(f"Starting CPG parsing using\n{_cfg_to_str(cfg.cmd.preprocess.parse)}")
    parse_cpgs(**cfg.cmd.preprocess.parse)


@preprocess.command(name="export")
def export_cli(overrides: List[str] = typer.Argument(default=None, help="override values for hydra")):
    """
    Export CPGs to JSON format.
    """
    cfg = get_cfg(to_remove=2)
    OmegaConf.to_object(cfg)
    log.info(f"Starting CPG export using\n{_cfg_to_str(cfg.cmd.preprocess.export)}")
    export_cpgs(**cfg.cmd.preprocess.export)


@preprocess.command(name="build-vocab")
def build_vocab_cli(overrides: List[str] = typer.Argument(default=None, help="override values for hydra")):
    """
    Build a vocabulary of node types.
    """
    cfg = get_cfg(to_remove=2)
    OmegaConf.to_object(cfg)
    log.info(f"Starting to build node type vocabulary using\n{_cfg_to_str(cfg.cmd.preprocess.build_vocab)}")
    build_vocab(**cfg.cmd.preprocess.build_vocab)


@preprocess.command(name="meta")
def meta_cli(overrides: List[str] = typer.Argument(default=None, help="override values for hydra")):
    """
    Extract metadata from raw input files.
    """
    cfg = get_cfg(to_remove=2)
    OmegaConf.to_object(cfg)
    log.info(f"Starting metadata extraction using\n{_cfg_to_str(cfg.cmd.preprocess.meta)}")
    PrimeVul.preprocess_meta(**cfg.cmd.preprocess.meta)


@preprocess.command(name="clone")
def clone_cli(overrides: List[str] = typer.Argument(default=None, help="override values for hydra")):
    """
    Clone repositories from raw input files.
    """
    cfg = get_cfg(to_remove=2)
    OmegaConf.to_object(cfg)
    log.info(f"Starting repository cloning using\n{_cfg_to_str(cfg.cmd.preprocess.clone)}")
    PrimeVul.clone_repos(**cfg.cmd.preprocess.clone)


"""
    Training
"""

train = typer.Typer()
app.add_typer(train, name="train")


@train.command(name="w2v")
def w2v_cli(overrides: List[str] = typer.Argument(default=None, help="override values for hydra")):
    """
    Train a word2vec model on a given corpus.
    """
    cfg = get_cfg(to_remove=2)
    OmegaConf.to_object(cfg)
    log.info(f"Starting W2V training using\n{_cfg_to_str(cfg.cmd.train.w2v)}")
    train_w2v(**cfg.cmd.train.w2v)


@train.command(name="start")
def train_cli(overrides: List[str] = typer.Argument(default=None, help="override values for hydra")):
    """
    Start training for a given model and dataset.
    """
    cfg = get_cfg(to_remove=2)
    OmegaConf.to_object(cfg)
    log.info(f"Starting new training run using\n{_cfg_to_str(cfg.cmd.train.train)}")
    run_train(**cfg.cmd.train.train)


"""
    CLI Setup
"""


def get_cfg(to_remove: int = 1, config_name="base_config", config_type=Config):
    remove_from_argv(to_remove)

    cs = ConfigStore.instance()
    cs.store(name=config_name, node=config_type)

    # hydra main decorator can't return anything (by design)
    global cfg
    set_global_cfg()
    return cfg


def remove_from_argv(n_args: int):
    assert n_args >= 1
    assert n_args < len(sys.argv)
    sys.argv = [sys.argv[0]] + sys.argv[n_args + 1:]


def _cfg_to_str(cfg: Config):
    c = OmegaConf.to_container(cfg, resolve=True)
    _resolve_dict(c)
    return OmegaConf.to_yaml(c)


def _resolve_dict(d):
    for k, v in list(d.items()):
        if isinstance(v, dict):
            _resolve_dict(v)
        elif isinstance(v, Path):
            v = str(v)
        elif isinstance(v, list):
            v = ', '.join(v)
        d[k] = v


@hydra.main(config_path="config", config_name="base", version_base=None)
def set_global_cfg(_cfg: Config):
    global cfg
    cfg = _cfg


def main():
    app()


if __name__ == "__main__":
    main()
