import click
import json
from pathlib import Path
from datasets import Dataset, DatasetDict


@click.command()
@click.option("--data", "-d", required=True, type=click.Path(dir_okay=True, file_okay=False, path_type=Path), help="Dataset directory.")
def create_hf_dataset(data):
    output_path = data.parent / "hf_full"
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = data / "train.jsonl"
    val_path = data / "val.jsonl"
    test_path = data / "test.jsonl"
    train_ds = [json.loads(line) for line in train_path.read_text().split("\n") if len(line) > 1]
    val_ds = [json.loads(line) for line in val_path.read_text().split("\n") if len(line) > 1]
    test_ds = [json.loads(line) for line in test_path.read_text().split("\n") if len(line) > 1]


    train_data = [{"texts": function["func"], "labels": function["target"]} for function in train_ds]
    val_data = [{"texts": function["func"], "labels": function["target"]} for function in val_ds]
    test_data = [{"texts": function["func"], "labels": function["target"]} for function in test_ds]

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
