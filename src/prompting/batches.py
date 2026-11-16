import argparse
import json
from pathlib import Path

from prompting.prompts import PROMPT_INST_COT, SYS_INST

# In our m size dataset, no function has more than 80_000 characters
MAX_CHARS = 100_000

# Maximum number of tokens that can be generated
MAX_TOKENS = 4_096


def create_request_from_sample(sample, dataset, model, inst, method, url):
    sample_id = sample["hf-idx"]
    request_id = f"{model}-{dataset}-{sample_id}"
    function_source_code = sample["texts"][:MAX_CHARS]
    prompt = inst.format(func=function_source_code)

    request  = {
        "custom_id": request_id,
        "method": method,
        "url": url,
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYS_INST},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": MAX_TOKENS,
        },
    }

    return request


def run_create_batch(dataset_dir: Path, model: str, size: str) -> None:
    dataset = dataset_dir.name
    input_fpath = dataset_dir / "prompting" / "raw-batches" / f"{dataset}-raw-batch-{size}.jsonl"
    output_dir = dataset_dir / "prompting" / "batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fpath = output_dir / f"{model}_{dataset}_{size}.jsonl"

    url = "/v1/chat/completions"
    method = "POST"
    inst = PROMPT_INST_COT

    raw_batch = (json.loads(line) for line in input_fpath.read_text().split("\n"))
    batch = (create_request_from_sample(sample, dataset, model, inst, method, url) for sample in raw_batch)

    batch_str = "\n".join(json.dumps(request) for request in batch)
    output_fpath.write_text(batch_str)


def main() -> None:
    p = argparse.ArgumentParser(description="Turn a raw batch into chain-of-thought requests for the OpenAI Batch API.")
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("model")
    p.add_argument("--size", default="s")
    args = p.parse_args()
    run_create_batch(args.dataset_dir, args.model, args.size)


if __name__ == "__main__":
    main()
