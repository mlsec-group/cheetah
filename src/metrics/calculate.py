import click
import os
import sys
from pathlib import Path

from mcpp import extract_code, METRICS
from datasets import load_from_disk
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.setrecursionlimit(25_000)

CHOICES = ["primevul", "primevul-xs", "diversevul-projects", "diversevul-random", "diversevul-xs"]

@click.command()
@click.option("--data-dir", required=True, type=click.Path(dir_okay=True, file_okay=False, path_type=Path))
@click.option("--dataset-name", required=True, type=click.Choice(CHOICES))
@click.option("--base_model", default=None)
@click.option("--num-cpus", default=8, type=int)
def main(data_dir, dataset_name, base_model, num_cpus):
    tokenizer = None
    if base_model is not None:
        if base_model == "Salesforce/codegen25-7b-multi_P":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi_P", trust_remote_code=True, revision="c936d25f73e74984748a5ba39151b6bc7a056e2e")
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        model_max_length = 1024 if base_model == "microsoft/unixcoder-base-nine" else 512

    dataset = load_from_disk(data_dir / dataset_name / "hf")

    def add_features(sample):
        try:
            text = sample["texts"]
            if tokenizer is not None:
                text = tokenizer.decode(tokenizer(text, truncation=True, max_length=model_max_length).input_ids)
            return extract_code(text)
        except:
            return {k: 0 for k in METRICS.keys()}

    dataset = dataset.map(add_features, batched=False, num_proc=num_cpus, load_from_cache_file=False)

    model_suffix = ""
    if base_model is not None:
        model_suffix = base_model.replace("/", "__")
    dataset.save_to_disk(data_dir / dataset_name / f"mcpp-{model_suffix}")


if __name__ == "__main__":
    main()
