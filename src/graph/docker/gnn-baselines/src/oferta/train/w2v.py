from typing import List
from pathlib import Path

from gensim.models import Word2Vec, KeyedVectors

from gnn-baselines.config.models import Word2VecConfig
from gnn-baselines.train.tokenize import NCSCorpus


def train_w2v(in_dir: Path, w2v_cfg: Word2VecConfig, ignore_tokens: List[str], model_path: Path):
    corpus = NCSCorpus(in_dir, ignore_tokens, w2v_cfg.workers)
    model = Word2Vec(sentences=corpus, **w2v_cfg)
    model_path.parent.mkdir(exist_ok=True, parents=True)
    model.wv.save(str(model_path))


def get_vocab(w2v_kv_file: Path):
    kv = KeyedVectors.load(str(w2v_kv_file))
    return kv.key_to_index
