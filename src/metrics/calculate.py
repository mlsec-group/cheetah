import click
import os
import sys
from pathlib import Path
import threading
import signal
from contextlib import contextmanager
import time

# Add src directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mcpp import extract_code, METRICS
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.setrecursionlimit(25_000)

CHOICES = ["primevul", "primevul-xs", "diversevul-projects", "diversevul-random", "diversevul-xs"]

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


@click.command()
@click.option("--data-dir", required=True, type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--dataset-name", required=True, type=click.Choice(CHOICES))
@click.option("--base_model", default=None)
@click.option("--truncate/--no-truncate", default=True)
@click.option("--num-cpus", default=256, type=int)
def main(data_dir, dataset_name, base_model, truncate, num_cpus):
    tokenizer = None
    if base_model is not None:
        if base_model == "Salesforce/codegen25-7b-multi_P":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi_P", trust_remote_code=True, revision="c936d25f73e74984748a5ba39151b6bc7a056e2e")
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        model_max_length = 1024 if base_model == "microsoft/unixcoder-base-nine" else 512
    if not os.path.exists(os.path.join(os.path.join(data_dir, dataset_name), "hf")):
        from dataset.dataset import load_data
        
        # Replicate the create_hf_dataset logic
        raw_path = data_dir / "raw"
        output_path = os.path.join(os.path.join(data_dir, dataset_name), "hf")
        os.makedirs(output_path)
        raw_data = os.listdir(raw_path)
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
            "test": val_dataset,
            "val": test_dataset
        })

        hf_dataset.save_to_disk(output_path)
    dataset = load_from_disk(data_dir / dataset_name / "hf")

    truncation_kwargs = {"truncation": False}
    if truncate:
        truncation_kwargs = {
            "truncation": True,
            "max_length": model_max_length,
        }
    timeout = 30
    def add_features(sample):
        result = {k: 0 for k in METRICS_ALIASES.keys()}
        try:
            with kill_after(timeout):
                text = sample["text"]
                if tokenizer is not None:
                    text = tokenizer.decode(tokenizer(text, **truncation_kwargs).input_ids)
                    result = extract_code(text)
                    result = {METRICS_ALIASES[k]: v for k, v in result.items()}
        except:
            pass
        return result

    dataset = dataset.map(add_features, batched=False, num_proc=num_cpus, load_from_cache_file=False)

    model_suffix = ""
    if base_model is not None:
        model_suffix = base_model.replace("/", "__")
    dataset.save_to_disk(data_dir / dataset_name / f"mcpp-{model_suffix}")


if __name__ == "__main__":
    main()
