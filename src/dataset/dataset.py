import click
import json
from pathlib import Path
from datasets import Dataset, DatasetDict


def load_data(fpath):
    samples = [
        json.loads(line) for line in fpath.read_text().split("\n") if len(line) > 1
    ]
    return [
        {"texts": function["func"], "labels": function["target"]} for function in samples
    ]


@click.command()
@click.option("--data", "-d", required=True, type=click.Path(dir_okay=True, file_okay=False, path_type=Path), help="Dataset directory.")
@click.option("--dataset-name", required=True, type=str, help="Dataset name.")
def create_hf_dataset(data, dataset_name):
    data = data / dataset_name
    raw_path = data / "raw"
    output_path = data / "hf"
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = raw_path / "primevul_train.jsonl"
    val_path = raw_path / "primevul_valid.jsonl"
    test_path = raw_path / "primevul_test.jsonl"
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


if __name__ == "__main__":
    create_hf_dataset()
