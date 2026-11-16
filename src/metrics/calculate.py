import argparse
import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_from_disk
from mcpp import extract_code
from transformers import AutoTokenizer

from dataset.dataset import load_data
from paths import model_slug, resolve_model_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.setrecursionlimit(25_000)

CHOICES = ["primevul", "primevul-lookaround"]

# The metrics as described in Table 4 in the paper and their aliases in mcpp
METRICS_ALIASES = {
    "S1": "s1",
    "S2": "s2",
    "S3": "s3",
    "S4": "s4",
    "S5": "V10",
    "C1": "C1",
    "C2": "C2",
    "C3": "C3",
    "C4": "C4",
    "C5": "V1",
    "C6": "V6",
    "C7": "V7",
    "C8": "V8",
    "C9": "x1",
    "C10": "x2",
    "C11": "x3",
    "C12": "x4",
    "M1": "m1",
    "M2": "m2",
    "M3": "V5",
    "T1": "t1",
    "T2": "t2",
    "T3": "t3",
}


@contextmanager
def kill_after(t):

  def raise_timeout_handler(signum, frame):
    raise TimeoutError

  orig_handler = signal.getsignal(signal.SIGALRM)
  signal.signal(signal.SIGALRM, raise_timeout_handler)
  try:
    signal.alarm(t)
    yield
  finally:
    signal.alarm(0)
    signal.signal(signal.SIGALRM, orig_handler)


def run_calculate(
    data_dir: Path,
    dataset_name: str,
    base_model: Optional[str],
    truncate: bool,
    num_cpus: Optional[int],
) -> None:
    tokenizer = None
    model_max_length = 512
    if base_model is not None:
        if base_model == "Salesforce/codegen25-7b-multi_P":
            tokenizer = AutoTokenizer.from_pretrained(
                "Salesforce/codegen25-7b-multi_P",
                trust_remote_code=True,
                revision="c936d25f73e74984748a5ba39151b6bc7a056e2e",
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(base_model))
        model_max_length = 1024 if base_model == "microsoft/unixcoder-base-nine" else 512
    if not os.path.exists(os.path.join(os.path.join(data_dir, dataset_name), "hf")):
        # Replicate the create_hf_dataset logic
        raw_path = data_dir / "raw"
        output_path = os.path.join(os.path.join(data_dir, dataset_name), "hf")
        os.makedirs(output_path)
        train_file = "primevul_train.jsonl"
        test_file = "primevul_test.jsonl"
        val_file = "primevul_valid.jsonl"
        for i in raw_path:
            if "train" in i.lower():
                train_file = i
            elif "val" in i.lower():
                val_file = i
            elif "test" in i.lower():
                test_file = i

        train_path = raw_path / train_file
        val_path = raw_path / val_file
        test_path = raw_path / test_file
        train_data = load_data(train_path)
        val_data = load_data(val_path)
        test_data = load_data(test_path)

        # Convert lists of dictionaries to Hugging Face Dataset objects
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)

        # Combine into a DatasetDict with splits
        hf_dataset = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        })

        hf_dataset.save_to_disk(output_path)
    dataset = load_from_disk(str(data_dir / dataset_name / "hf"))
    split_sizes = {split: len(dataset[split]) for split in dataset}
    total_examples = sum(split_sizes.values())
    output_path = data_dir / dataset_name / f"mcpp-{model_slug(base_model) if base_model else ''}".rstrip("-")
    print(
        f"[metrics.calculate] dataset={dataset_name} splits={split_sizes} total={total_examples} "
        f"truncate={truncate} num_proc={num_cpus} output={output_path}",
        flush=True,
    )

    truncation_kwargs = {"truncation": False}
    if truncate:
        truncation_kwargs = {
            "truncation": True,
            "max_length": model_max_length,
        }
    timeout = 30

    def add_features(sample):
        result = {k: 0.0 for k in METRICS_ALIASES.keys()}
        try:
            with kill_after(timeout):
                text = sample["texts"]
                if tokenizer is not None:
                    text = tokenizer.decode(tokenizer(text, **truncation_kwargs).input_ids)
                    result = extract_code(text)
                    result = {k: float(result[METRICS_ALIASES[k]]) for k in METRICS_ALIASES.keys()}
        except TimeoutError:
            pass
        except Exception:
            import traceback
            traceback.print_exc()
        return result

    print(f"[metrics.calculate] starting mcpp map over {total_examples} examples", flush=True)
    dataset = dataset.map(add_features, batched=False, num_proc=num_cpus, load_from_cache_file=False)
    print(f"[metrics.calculate] mcpp map finished, saving to {output_path}", flush=True)

    dataset.save_to_disk(output_path)
    print(f"[metrics.calculate] done", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract MCPP metrics.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--truncate/--no-truncate", dest="truncate", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--num-cpus", type=int, default=None)
    args = p.parse_args()
    if args.dataset_name not in CHOICES:
        p.error(f"dataset-name must be one of {CHOICES}")
    run_calculate(
        args.data_dir,
        args.dataset_name,
        args.base_model,
        args.truncate,
        args.num_cpus,
    )


if __name__ == "__main__":
    main()
