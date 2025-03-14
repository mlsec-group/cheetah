import torch

from gnn-baselines.models.simple import SimpleModel
from gnn-baselines.models.linear_mcpp import LinearMCPP
from gnn-baselines.models.devign import Devign
from gnn-baselines.models.reveal import Reveal
from gnn-baselines.models.llm import CodeT5, UniXcoder


MODELS = {
    "simple": SimpleModel,
    "linear_mcpp": LinearMCPP,
    "devign": Devign,
    "reveal": Reveal,
    "codet5": CodeT5,
    "unixcoder": UniXcoder
}

LLMS = {
    "codet5": CodeT5,
    "unixcoder": UniXcoder
}


def init_model(name: str, class_ratio, **kwargs):
    if name not in MODELS.keys():
        raise ValueError(f"Unknown model: {name}. Must be one of {sorted(MODELS.keys())}")

    model = MODELS[name](name, class_ratio, **kwargs)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def get_feature_set(name: str):
    if name not in MODELS.keys():
        raise ValueError(f"Unknown model: {name}. Must be one of {sorted(MODELS.keys())}")
    return MODELS[name]._feature_set


def get_all_feature_sets():
    fsets = {}
    for model in MODELS.values():
        fsets.add(model._feature_set)
    return sorted(fsets)
