import argparse
import json
import re
from datetime import datetime
from hashlib import md5
from pathlib import Path

from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset, load_from_disk
from loguru import logger
from transformers import AutoTokenizer

from graph.node_types import DEFAULT_X_TYPES


def init(raw_path: Path, hf_cache: Path, base_model: str, model_max_length: int, debug: bool = False):
    ds = DatasetDict(
        {
            split: load_dataset("json", data_files=str(raw_path / f"{split}.jsonl"), split="train", cache_dir=hf_cache)
            for split in ["train", "val", "test"]
        }
    )
    ds = ds.rename_column("target", "label")
    ds = ds.rename_column("func", "text")
    if debug:
        ds = to_debug(ds)
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=hf_cache)
    tokenizer.model_max_length = model_max_length

    # Let's misuse some standard keys to store metadata
    metadata = DatasetInfo(
        description=base_model,
        citation=str(hf_cache),  # Convert Path to string for serialization
        homepage=model_max_length,
        license=datetime.now().isoformat(),
    )

    for split in ds.keys():
        ds[split].info.update(metadata)

    return ds, tokenizer


def load(ds_path: Path, seed: int = 42, subsample_factor: float = 0.9):
    ds = load_from_disk(str(ds_path))
    NUMBER_TRAIN_SAMPLES = int(len(ds["train"]) * subsample_factor)
    logger.info(f"Selecting {NUMBER_TRAIN_SAMPLES} of {len(ds['train'])} samples for training")
    train_set = ds["train"].shuffle(seed=seed, load_from_cache_file=False).select(range(NUMBER_TRAIN_SAMPLES))
    ds["train"] = train_set

    base_model = ds["train"].info.description
    hf_cache = Path(ds["train"].info.citation)
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=hf_cache)
    tokenizer.model_max_length = int(ds["train"].info.homepage)
    try:
        with open(ds_path / "node_type_vocab.json", "r") as f:
            node_type_vocab = json.load(f)
    except FileNotFoundError:
        node_type_vocab = DEFAULT_X_TYPES
    return ds, tokenizer, node_type_vocab


def subsample(ds: Dataset, ratio: float, seed: int = 42):
    assert 0 < ratio <= 1
    return ds.shuffle(seed=seed).select(range(int(len(ds) * ratio)))


def to_debug(ds: Dataset, seed: int = 42):
    ds["train"] = subsample(ds["train"], 0.01, seed)
    ds["val"] = subsample(ds["val"], 0.01, seed)
    ds["test"] = subsample(ds["test"], 0.01, seed)
    return ds


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer):
    logger.info(f"Tokenizing datasets with maximum length {tokenizer.model_max_length}")
    tokenize_with_tokenizer = lambda s: tokenize(s, tokenizer, tokenizer.model_max_length)  # noqa: E731
    dataset = dataset.map(
        tokenize_with_tokenizer,
        batched=True,
        num_proc=None,
        load_from_cache_file=False,
    )
    return dataset


def tokenize(sample, tokenizer, max_length):
    tokenized = tokenizer(sample["text"], truncation=True, max_length=max_length, return_overflowing_tokens=True)
    num_truncated_tokens = [0] * len(sample["label"])
    input_lengths = []
    input_ids = []
    attention_mask = []
    last = -1
    for idx, enc in zip(tokenized["overflow_to_sample_mapping"], tokenized.encodings):
        if idx != last:
            input_ids.append(enc.ids)
            attention_mask.append(enc.attention_mask)
            input_lengths.append(len(enc.ids))
        else:
            num_truncated_tokens[idx] += len(enc.ids)
        last = idx
    tokenized["input_ids"] = input_ids
    tokenized["num_truncated_tokens"] = num_truncated_tokens
    tokenized["attention_mask"] = attention_mask
    tokenized["input_lengths"] = input_lengths
    if "overflow_to_sample_mapping" in tokenized:
        del tokenized["overflow_to_sample_mapping"]
    sample.update(tokenized)
    return sample


RE_NORMALIZE_CODE = re.compile(r"\s+")


def normalize_code(code: str) -> str:
    """Remove all whitespace characters from the code."""
    return RE_NORMALIZE_CODE.sub("", code)


def primevul_hash(func: str) -> int:
    norm_code = normalize_code(func)
    h = md5(norm_code.encode()).hexdigest()
    return int(h, 16)


def preprocess(raw_path: Path, ds_path: Path, hf_cache: Path, base_model: str, model_max_length: int, debug: bool = False):
    ds, tokenizer = init(raw_path, hf_cache, base_model, model_max_length, debug)
    ds = tokenize_dataset(ds, tokenizer)
    ds.save_to_disk(ds_path)


def write_sources(ds_path: Path, output_dir: Path):
    ds, _, _ = load(ds_path)
    for split in ds.keys():
        for sample in ds[split]:
            subdir = output_dir / split / sample.get("project", "default")
            subdir.mkdir(parents=True, exist_ok=True)
            code = sample["text"]
            with open(subdir / f"{sample['idx']}.c", "w") as f:
                f.write(code)


def add_graphs(ds_path: Path, graph_root: Path, out_path: Path):
    from graph.datasets.graphs import get_node_type_vocab, has_graph, to_pyg

    ds, _, _ = load(ds_path)
    node_type_vocab = get_node_type_vocab(ds, graph_root)

    # only keep samples with graphs
    for split in ds.keys():
        ds[split] = ds[split].filter(lambda x: has_graph(x, graph_root, split))

    # map json graphs to pytorch geometric format
    for split in ds.keys():
        ds[split] = ds[split].map(
            to_pyg,
            fn_kwargs={
                "graph_root": graph_root,
                "split": split,
                "node_type_vocab": node_type_vocab,
                "base_model": ds[split].info.description,
                "hf_cache": ds[split].info.citation,
                "model_max_length": ds[split].info.homepage,
            },
            batched=True,
            num_proc=None,
            load_from_cache_file=False,
        )
    ds.save_to_disk(out_path)
    with open(out_path / "node_type_vocab.json", "w") as f:
        json.dump(node_type_vocab, f, indent=4)


def main() -> None:
    p = argparse.ArgumentParser(description="PrimeVul dataset CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preprocess", help="Build HF dataset on disk.")
    sp.add_argument("raw_path", type=Path)
    sp.add_argument("ds_path", type=Path)
    sp.add_argument("hf_cache", type=Path)
    sp.add_argument("base_model")
    sp.add_argument("model_max_length", type=int)
    sp.add_argument("--debug", action="store_true")

    sw = sub.add_parser("write-sources", help="Extract .c sources for Joern.")
    sw.add_argument("ds_path", type=Path)
    sw.add_argument("output_dir", type=Path)

    sa = sub.add_parser("add-graphs", help="Attach exported graphs to HF dataset.")
    sa.add_argument("ds_path", type=Path)
    sa.add_argument("graph_root", type=Path)
    sa.add_argument("out_path", type=Path)

    args = p.parse_args()
    if args.cmd == "preprocess":
        preprocess(
            args.raw_path,
            args.ds_path,
            args.hf_cache,
            args.base_model,
            args.model_max_length,
            debug=args.debug,
        )
    elif args.cmd == "write-sources":
        write_sources(args.ds_path, args.output_dir)
    else:
        add_graphs(args.ds_path, args.graph_root, args.out_path)


if __name__ == "__main__":
    main()
