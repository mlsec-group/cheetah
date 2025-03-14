from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.data import Data
from optuna.trial import Trial
from transformers import AutoModel

from gnn-baselines.models.base import Model, ClassificationHead


def init_embedder(llm_specifier, llm_cache):
    return AutoModel.from_pretrained(llm_specifier, trust_remote_code=True, cache_dir=llm_cache)


class LLM(Model, ABC):
    _feature_set = "llm"
    huggingface_name = None
    tokenizer_max_len = None
    embedder_max_len = None
    emb_dim = None

    def __init__(self, name, class_ratio, test_predictions_path, input_dim, hidden_dim, dropout, llm_cache, **hparams):
        super().__init__(name, LLM._feature_set, class_ratio, test_predictions_path, **hparams)
        self.llm_cache = llm_cache
        self._classifier = ClassificationHead(input_dim, hidden_dim, self.out_dim, dropout).classifier
        self.save_hyperparameters({"input_dim": input_dim, "hidden_dim": hidden_dim, "dropout": dropout})

    @property
    @abstractmethod
    def embedder(self):
        raise NotImplementedError()

    @property
    def classifier(self):
        return self._classifier

    @property
    @abstractmethod
    def max_len(self):
        raise NotImplementedError()


class CodeT5(LLM):
    huggingface_name = "Salesforce/codet5p-110m-embedding"
    tokenizer_max_len = 100_000
    embedder_max_len = 512  # model.config.n_positions
    emb_dim = 256  # model(model.dummy_inputs["decoder_input_ids"]).shape[-1]

    def __init__(self, *args, **kwargs):
        for key, value in list(kwargs.items()):
            if value is None:
                del kwargs[key]
        super().__init__(*args,
                         input_dim=CodeT5.emb_dim, huggingface_name=CodeT5.huggingface_name,
                         embedder_max_len=CodeT5.embedder_max_len,
                         **kwargs)
        self._embedder = CodeT5Embedder(
            self.name,
            init_embedder(self.huggingface_name, self.embedder_max_len),
            self.llm_cache
        )

    @property
    def max_len(self):
        return self.embedder.config.n_positions

    @property
    def embedder(self):
        return self._embedder

    @staticmethod
    def suggest_hparams(trial: Trial):
        hparams = {
            "hidden_dim": trial.suggest_int("models.codet5.hidden_dim", 64, 512),
            "dropout": trial.suggest_float("models.codet5.dropout", 0.0, 1.0),
        }
        return hparams


class CodeT5Embedder(nn.Module):
    """Embedding Module for CodeT5."""

    def __init__(self, name, model, embedder_max_len):
        super().__init__()
        self.name = name
        self.model = model
        self.embedder_max_length = embedder_max_len

    def forward(self, data: Data) -> torch.Tensor:
        vecs = []
        for d in data.to_data_list():
            tokens = d[self.name][:self.embedder_max_length]
            vecs.append(self.model(tokens.view(1, -1))[0])
        return torch.stack(vecs)


class UniXcoder(LLM):
    huggingface_name = "microsoft/unixcoder-base-nine"
    tokenizer_max_len = 100_000
    num_reserved_positions = 2
    embedder_max_len = 1026 - num_reserved_positions  # model.config.max_position_embeddings
    emb_dim = 768  # model.config.hidden_size

    def __init__(self, *args, **kwargs):
        for key, value in list(kwargs.items()):
            if value is None:
                del kwargs[key]
        super().__init__(*args,
                         input_dim=UniXcoder.emb_dim, huggingface_name=UniXcoder.huggingface_name,
                         embedder_max_len=UniXcoder.embedder_max_len,
                         **kwargs)
        self._embedder = UniXcoderEmbedder(
            self.name,
            init_embedder(self.huggingface_name, self.llm_cache),
            self.embedder_max_len
        )

    @property
    def max_len(self):
        return self.embedder.config.max_position_embeddings - self._num_reserved_positions

    @property
    def embedder(self):
        return self._embedder

    @staticmethod
    def suggest_hparams(trial: Trial):
        hparams = {
            "hidden_dim": trial.suggest_int("models.unixcoder.hidden_dim", 64, 512),
            "dropout": trial.suggest_float("models.unixcoder.dropout", 0.0, 1.0),
        }
        return hparams


class UniXcoderEmbedder(nn.Module):
    """Embedding Module for UniXcoder."""

    def __init__(self, name, model, embedder_max_len):
        super().__init__()
        self.name = name
        self.model = model
        self.embedder_max_length = embedder_max_len

    def forward(self, data: Data) -> torch.Tensor:
        vecs = []
        for d in data.to_data_list():
            tokens = d[self.name][:self.embedder_max_length]
            vecs.append(self.model(tokens.view(1, -1)).pooler_output[0])
        return torch.stack(vecs)
