from pathlib import Path
from json import JSONDecodeError

import torch
from loguru import logger as log
from rich.progress import track

import joblib.externals.loky

joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(3e10)

from gnn-baselines.config.schema import TokenizerConfig
from gnn-baselines.datasets.common import data_from_json
from gnn-baselines.datasets.base import Dataset
from gnn-baselines.datasets.feature_set import FeatureSet
from gnn-baselines.train.w2v import get_vocab
from gnn-baselines.models.base import NodeTypeEmbedding
from gnn-baselines.datasets.metrics import FeatureMean, FeatureStd, LabelDist
from gnn-baselines.util import ParallelProgress


class DiverseVul(Dataset):
    _name = "diversevul"
    _num_classes = 2

    def __init__(
        self,
        root: Path,
        feature_set: str,
        processed_file: Path,
        node_type_vocab: Path,
        w2v_kv_file: Path,
        tokenizer_cfg: TokenizerConfig,
        transform=None,
        pre_transform=None,
        llms=None,
        llm_cache=None,
        interim_dir=None,
        jobs=1,
        seed=None,
    ):
        temp_dir = None
        if processed_file is None:
            processed_file = root / f"{feature_set}_cache.bin"

        super().__init__(
            self._name,
            self._num_classes,
            feature_set,
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            seed=seed,
        )
        self._processed_file = processed_file

        self.root = root
        self._data_files = sorted(root.glob("**/*.json"))
        self.node_type_vocab = node_type_vocab
        self.node_type_embedding = NodeTypeEmbedding(self.node_type_vocab, one_hot=True)
        self.w2v_kv_file = w2v_kv_file
        self.tokenizer_cfg = tokenizer_cfg
        self.llms = llms
        self.llm_cache = llm_cache
        self.interim_dir = interim_dir
        self.jobs = jobs

        if not self._processed_file.exists():
            self._process()
        self.data, self.slices, self.skipped_idx, self.stats = torch.load(self._processed_file)
        self._data_files = [f for i, f in enumerate(self._data_files) if i not in self.skipped_idx]
        if temp_dir is not None:
            temp_dir.cleanup()

        # WORKAROUND: class ratio only for train set
        class_ratio = torch.zeros(self.num_classes)
        for in_file in self.data_files:
            if "train" in in_file.parts:
                class_ratio[int(in_file.stem.split("_")[0])] += 1
        class_ratio /= class_ratio.sum()
        self.stats["class_ratio"] = class_ratio

    @property
    def processed_file_names(self):
        return [self.processed_file.name]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data_files(self):
        return self._data_files

    def _process(self):
        vocab = get_vocab(self.w2v_kv_file)
        feature_set_kwargs = {
            "interim_dir": self.interim_dir,
            "tokenizer_cfg": self.tokenizer_cfg,
            "node_type_embedding": self.node_type_embedding,
            "llm_cache": self.llm_cache,
            "vocab": vocab,
            "w2v_kv_file": self.w2v_kv_file
        }
        # initialize feature set once to run optional extractor preparation steps
        # (e.g. downloading pre-trained LLMs) sequentially
        feature_set = FeatureSet.by_name(self.feature_set, **feature_set_kwargs)

        data_list, skipped_idx = self._json_to_data(**feature_set_kwargs)
        stats = self._calc_ds_stats(data_list, feature_set)

        data, slices = self.collate(data_list)
        self._processed_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save((data, slices, skipped_idx, stats), self._processed_file)

    def _json_to_data(self, **feature_set_kwargs):
        samples = []
        skipped_idx = []

        log.info(f"starting extraction for feature set {self.feature_set}")
        with ParallelProgress(self.jobs, task=self.feature_set) as progress:
            args = [(in_file, self.feature_set, feature_set_kwargs) for in_file in self.data_files]
            samples = progress.run(process_single, args)

        skipped_idx = [i for i, sample in enumerate(samples) if sample is None]
        samples = [sample for sample in samples if sample is not None]
        skipped_idx = torch.tensor(skipped_idx, dtype=torch.long)
        return samples, skipped_idx

    def _calc_ds_stats(self, data_list, feature_set: FeatureSet):
        # TODO: calc only over training set
        mean_stats = [FeatureMean(key) for key in feature_set.stat_keys]
        log.info("starting first pass (feature mean)")
        for data in track(data_list, description="[green]Calculating feature mean..."):
            for mean_stat in mean_stats:
                mean_stat.update(data)
        means = {stat.key: stat.compute() for stat in mean_stats}
        label_dist = LabelDist()
        for in_file in self.data_files:
            if "train" in in_file.parts:
                label_dist.update(data)
        class_ratio = label_dist.compute()

        std_stats = [FeatureStd(key, means[key]) for key in feature_set.stat_keys]
        if len(std_stats) > 0:
            log.info("starting second pass (feature std)")
            for data in track(data_list, description="[green]Calculating feature std..."):
                for std_stat in std_stats:
                    std_stat.update(data)
        stds = {stat.key: stat.compute() for stat in std_stats}
        return {
            "class_ratio": class_ratio,
            "mean": means,
            "std": stds
        }


def process_single(in_file: Path, feature_set: str, feature_set_kwargs):
    try:
        data = data_from_json(in_file, feature_set, **feature_set_kwargs)
    except JSONDecodeError:
        log.warn(f"broken or empty sample {in_file}")
        return None
    return data
    """
    if feature_set != "devign" or data.num_nodes >= 10:
        return data
    else:
        log.info(f"Skipping {in_file} because it has {data.num_nodes} < 10 or > 1000 nodes")
        # raise "Sample should have been filtered out during preprocessing already"
        return None
    """
