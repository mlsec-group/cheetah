import click
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk


@click.command()
@click.option("--data", "-d", required=True, type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path), help="Data directory.")
@click.option("--force/--no-force", default=False, type=bool, help="Force re-creation if dataset was already processed.")
def hf_to_dir(data, force):
    hf_dir = data /  "hf"
    output_dir = data / "sast" / "sources"

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
