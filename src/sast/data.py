import argparse
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm


def run_hf_to_dir(data: Path, force: bool) -> None:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run_hf_to_dir(args.data, args.force)


if __name__ == "__main__":
    main()
