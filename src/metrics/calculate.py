import click
import os
import sys
from pathlib import Path

# Add src directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from mcpp import extract_code, METRICS
from datasets import load_from_disk, Dataset, DatasetDict
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
