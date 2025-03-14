from pathlib import Path

from transformers import AutoTokenizer

from gnn-baselines.datasets.extractors.base import Extractor
from gnn-baselines.models import LLMS


# for usage in multi-processing (avoid to reload them in persistent workers)
_tokenizer = {}


def init_tokenizer(llm_specifier, llm_cache):
    return AutoTokenizer.from_pretrained(llm_specifier, trust_remote_code=True, cache_dir=llm_cache)


class LLMExtractor(Extractor):

    def __init__(self, interim_dir: Path, feat_name: str, llm_name: str, llm_cache: Path):
        super().__init__([feat_name], interim_dir)
        self.llm_name = llm_name
        self.llm_cache = llm_cache
        self.feat_name = feat_name
        self.extract_funcs = [self.extract_llm]

    @property
    def stat_keys(self):
        return []

    def extract_llm(self, in_file, graph, code):
        return _extract_llm(in_file, graph, code, self.feat_name, self.llm_cache)


def _extract_llm(in_file, graph, code, feat_name, llm_cache):
    global _tokenizer

    if feat_name not in _tokenizer:
        _tokenizer[feat_name] = init_tokenizer(LLMS[feat_name].huggingface_name, llm_cache)

    # TODO truncation is actually quite bad
    tokenizer_max_length = min(_tokenizer[feat_name].model_max_length, LLMS[feat_name].tokenizer_max_len)
    tokens = _tokenizer[feat_name].encode(code, return_tensors="pt", max_length=tokenizer_max_length, truncation=True)[0]
    features = {feat_name: tokens}
    return features
