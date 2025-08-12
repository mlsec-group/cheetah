import click
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk

CHOICES = ["primevul", "primevul-xs", "diversevul-projects", "diversevul-random", "diversevul-xs"]

@click.command()
@click.option("--data", "-d", required=True, type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path), help="Data directory.")
@click.option("--dataset-name", required=True, type=click.Choice(CHOICES))
@click.option("--force/--no-force", default=False, type=bool, help="Force re-creation if dataset was already processed.")
def hf_to_dir(data, dataset_name , force):
    hf_dir = data / dataset_name/ "hf"
    output_dir = data / dataset_name/ "sast" / "sources"

    if output_dir.exists() and not force:
        return
    dataset = load_from_disk(hf_dir)
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"[+] {split}")
        for idx, element in tqdm(enumerate(dataset[split]), total=len(dataset[split])):
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            fpath = split_dir / f"{idx}.cpp"
            fpath.write_text(element["texts"])

if __name__ == "__main__":
    hf_to_dir()
