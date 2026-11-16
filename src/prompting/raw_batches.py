import argparse
import json
from pathlib import Path
from random import seed, shuffle

from datasets import load_from_disk

# Target sizes of the stratified subsets. They are drawn one after the other
# from the same random stream, so the order must not change.
SIZES = {"full": 100_000_000, "m": 10000, "s": 2000, "xs": 100, "xxs": 10}


def create_raw_batch(data, target_size, output_fpath):
    imbalance_factor = len(data["vuln"]) / len(data["novuln"])
    vuln_size = int(target_size * imbalance_factor)
    novuln_size = target_size - vuln_size

    shuffle(data["vuln"])
    shuffle(data["novuln"])

    vuln_data = data["vuln"][:vuln_size]
    novuln_data = data["novuln"][:novuln_size]
    output_data = vuln_data + novuln_data

    output_data_str = "\n".join(json.dumps(dp) for dp in output_data)
    output_fpath.write_text(output_data_str)


def run_create_raw_batches(dataset_dir: Path) -> None:
    seed(1337)

    dataset = load_from_disk(str(dataset_dir / "hf"))
    data = {"vuln": [], "novuln": []}

    for idx, dp in enumerate(dataset["test"]):
        dp["hf-idx"] = idx
        if dp["labels"] == 0:
            data["novuln"].append(dp)
        else:
            data["vuln"].append(dp)

    output_dir = dataset_dir / "prompting" / "raw-batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    for size, target_size in SIZES.items():
        create_raw_batch(data, target_size, output_dir / f"{dataset_dir.name}-raw-batch-{size}.jsonl")


def main() -> None:
    p = argparse.ArgumentParser(description="Draw stratified subsets of the test split for the prompting experiment.")
    p.add_argument("dataset_dir", type=Path)
    args = p.parse_args()
    run_create_raw_batches(args.dataset_dir)


if __name__ == "__main__":
    main()
