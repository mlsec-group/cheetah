from typing import List
from pathlib import Path

from gnn-baselines.datasets.extractors.base import Extractor
from gnn-baselines.datasets.extractors import LabelExtractor, GraphExtractor, LLMExtractor, McppExtractor
from gnn-baselines.models import LLMS


class FeatureSet(object):
    def __init__(self, name: str, extractors: List[Extractor]):
        self.name = name
        self.extractors = extractors

    @property
    def keys(self) -> List[str]:
        keyset = set()
        for extractor in self.extractors:
            keyset |= set(extractor.keys)
        return sorted(keyset)

    @property
    def stat_keys(self) -> List[str]:
        keyset = set()
        for extractor in self.extractors:
            keyset |= set(extractor.stat_keys)
        return sorted(keyset)

    @staticmethod
    def by_name(name, **kwargs):
        feature_sets = [DevignFeatures, LLMFeatures, McppFeatures]
        for FeatureSetCls in feature_sets:
            if FeatureSetCls._name == name:
                return FeatureSetCls(**kwargs)
        raise ValueError(f"unknown feature set {name}. must be one of {[fs._name for fs in feature_sets]}")


class DevignFeatures(FeatureSet):
    _name = "devign"

    def __init__(self, interim_dir: Path, tokenizer_cfg, vocab, node_type_embedding, w2v_kv_file, **unused_kwargs):
        extractors = [
            LabelExtractor(interim_dir),
            GraphExtractor(interim_dir, tokenizer_cfg, vocab, node_type_embedding, w2v_kv_file),
        ]
        super().__init__(DevignFeatures._name, extractors)


class LLMFeatures(FeatureSet):
    _name = "llm"

    def __init__(self, interim_dir: Path, llm_cache: Path, **unused_kwargs):
        extractors = [LabelExtractor(interim_dir)]
        for name, llm_cls in LLMS.items():
            extractors.append(LLMExtractor(interim_dir, name, llm_cls.huggingface_name, llm_cache))
        super().__init__(LLMFeatures._name, extractors)


class McppFeatures(FeatureSet):
    _name = "mcpp"

    def __init__(self, interim_dir: Path, **unused_kwargs):
        extractors = [
            LabelExtractor(interim_dir),
            McppExtractor(interim_dir)
        ]
        super().__init__(McppFeatures._name, extractors)
