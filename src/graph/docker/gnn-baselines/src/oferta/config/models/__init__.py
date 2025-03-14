from typing import List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimpleConfig:
    input_dim: int
    hidden_dim: int
    dropout: float


@dataclass
class LinearMCPPConfig:
    input_dim: int
    mcpp_features: List[str]
    hidden_dim: int
    dropout: float


@dataclass
class DevignConfig:
    input_dim: int
    graph_embed_size: int
    num_timesteps: int
    dropout: float


@dataclass
class RevealConfig:
    input_dim: int
    hidden_dim: int
    ggnn_layers: int
    graph_embed_size: int
    extractor_layers: int
    dropout: float


@dataclass
class CodeT5Config:
    hidden_dim: int
    dropout: float
    llm_cache: Path


@dataclass
class UniXcoderConfig:
    hidden_dim: int
    dropout: float
    llm_cache: Path


@dataclass
class Word2VecConfig:
    min_count: int
    seed: int
    vector_size: int
    max_vocab_size: int
    workers: int
    sg: int


@dataclass
class ModelConfig:
    w2v: Word2VecConfig
    simple: SimpleConfig
    linear_mcpp: LinearMCPPConfig
    devign: DevignConfig
    reveal: RevealConfig
    codet5: CodeT5Config
    unixcoder: UniXcoderConfig
