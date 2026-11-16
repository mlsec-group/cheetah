from pathlib import Path

from transformers import AutoConfig, AutoModel

from graph.gnn.devign import Devign
from graph.gnn.dummy import DummyModel
from graph.gnn.reveal import Reveal

MODELS = (Reveal, Devign, DummyModel)


for model_class in MODELS:
    AutoConfig.register(model_class.config_class.model_type, model_class.config_class)
    AutoModel.register(model_class.config_class, model_class)


def load_model(ckpt_path: str, hf_cache: Path = None):
    model = AutoModel.from_pretrained(ckpt_path, cache_dir=hf_cache)
    return model


def init_model(model_name: str, node_type_vocab: dict = None, hf_cache: Path = None):
    for model_class in MODELS:
        if model_class.config_class.model_type == model_name:
            return model_class(model_class.config_class(x_types=list(node_type_vocab.keys())))
    raise ValueError(f"Model {model_name} not found. Available models: {[m.config_class.model_type for m in MODELS]}")
