import argparse
import os
import pickle
from pathlib import Path
from time import sleep

from openai import OpenAI


def submit_batch(client, batch_fpath, batch_id, create_new, cache_dir):
    # Upload batch
    print(f"[+] Upload batch {batch_id}")
    batch_file_cache_fpath = cache_dir / f"batch_{batch_id}_file_cache.pickle"
    print(batch_file_cache_fpath)
    if batch_file_cache_fpath.exists() and not create_new:
        batch_input_file = pickle.loads(batch_file_cache_fpath.read_bytes())
    else:
        batch_input_file = client.files.create(
            file=open(str(batch_fpath), "rb"),
            purpose="batch"
        )
        batch_file_cache_fpath.write_bytes(pickle.dumps(batch_input_file))

    # Create batch job
    print(f"[+] Create batch job {batch_id}")
    batch_input_file_id = batch_input_file.id
    batch_cache_fpath = cache_dir / f"batch_{batch_id}_cache.pickle"
    if batch_cache_fpath.exists() and not create_new:
        batch = pickle.loads(batch_cache_fpath.read_bytes())
    else:
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": batch_fpath.stem,
            }
        )
        batch_cache_fpath.write_bytes(pickle.dumps(batch))

    return batch


def download_results(client, batch, output_fpath):
    file_response = client.files.content(str(batch.output_file_id))
    output_fpath.write_text(file_response.text)


def submit(batch_fpath, repititions, client, output_dir, cache_dir, create_new):
    batches = []

    print("[+] Submit batches")
    for batch_no in range(repititions):
        batch_id = f"{batch_fpath.stem}-{batch_no}"
        batch = submit_batch(client, batch_fpath, batch_id, create_new=create_new, cache_dir=cache_dir)
        batches.append(batch)

    print("[+] Wait for batches to complete")
    while True:
        completed = 0
        for batch_no, batch in enumerate(batches):
            batch_id = f"{batch_fpath.stem}-{batch_no}"
            batch = client.batches.retrieve(batch.id)

            if batch.status == "completed":
                print(f"[+] Get results for {batch_id}")
                output_fpath = output_dir / f"{batch_id}.jsonl"
                download_results(client, batch, output_fpath)
                print("[+] -> Done")
                completed += 1
            elif batch.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} ended with status {batch.status}: {batch.errors}")

        if completed == len(batches):
            break

        sleep(60)


def run_submit(dataset_dir: Path, model: str, size: str, repititions: int, create_new: bool) -> None:
    prompting_dir = dataset_dir / "prompting"
    batch_fpath = prompting_dir / "batches" / f"{model}_{dataset_dir.name}_{size}.jsonl"
    output_dir = prompting_dir / "out"
    cache_dir = prompting_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    submit(batch_fpath, repititions, client, output_dir, cache_dir, create_new)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Submit a batch to the OpenAI Batch API several times and download the results. "
        "Submitted batches are cached, so an interrupted run can be resumed."
    )
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("model")
    p.add_argument("--size", default="s")
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--create-new", action="store_true", help="Ignore cached uploads and batch jobs")
    args = p.parse_args()
    run_submit(args.dataset_dir, args.model, args.size, args.repetitions, args.create_new)


if __name__ == "__main__":
    main()
