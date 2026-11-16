import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from paths import model_slug, resolve_model_path


def tokenize(sample, tokenizer, truncation_length):
    return tokenizer(sample["texts"], truncation=True, max_length=truncation_length)


def run_add_embeddings(dataset_dir: Path, base_model: str, uid: str) -> None:
    slug = model_slug(base_model)
    run_dir = dataset_dir / "hf-model" / slug / str(uid)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints directory: {run_dir}")

    checkpoints = [
        p for p in run_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-") and p.name.split("-")[1].isnumeric()
    ]
    assert checkpoints, f"No checkpoints under {run_dir}"
    smallest_checkpoint = min(checkpoints, key=lambda p: int(p.name.split("-")[1]))
    embedded_path = smallest_checkpoint / "embedded"
    if embedded_path.exists():
        print(f"[add_embeddings] {embedded_path} already exists, skipping", flush=True)
        return

    if base_model == "Salesforce/codegen25-7b-multi_P":
        tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/codegen25-7b-multi_P",
            trust_remote_code=True,
            revision="c936d25f73e74984748a5ba39151b6bc7a056e2e",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(base_model))
    if base_model in ["Salesforce/codegen25-7b-multi_P", "bigcode/starcoder2-7b"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        smallest_checkpoint, torch_dtype=torch.float16, device_map="auto"
    )
    if base_model in ["Salesforce/codegen25-7b-multi_P", "bigcode/starcoder2-7b"]:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    dataset = load_from_disk(str(dataset_dir / "hf"))
    model_max_length = 1024 if base_model == "microsoft/unixcoder-base-nine" else 512
    print(
        f"[add_embeddings] dataset={dataset_dir} uid={uid} checkpoint={smallest_checkpoint.name} "
        f"max_length={model_max_length}",
        flush=True,
    )

    columns_to_remove = [c for c in dataset["train"].column_names if "label" not in c]
    print(f"[add_embeddings] tokenize map, removing {columns_to_remove}", flush=True)
    dataset = dataset.map(
        tokenize, batched=True, num_proc=None,
        remove_columns=columns_to_remove, load_from_cache_file=False,
        fn_kwargs=dict(tokenizer=tokenizer, truncation_length=model_max_length),
    )

    print("[add_embeddings] starting embedding map", flush=True)
    device = next(model.parameters()).device
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    @torch.no_grad()
    def create_embeddings(sample):
        inputs = collator(sample)
        inputs = inputs.to(device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        if base_model not in ["Salesforce/codegen25-7b-multi_P", "bigcode/starcoder2-7b"]:
            embeddings = hidden_states[:, 0, :]
            embeddings = model.classifier.dense(embeddings)
        else:
            input_ids = inputs["input_ids"]
            batch_size = input_ids.shape[0]
            sequence_lengths = torch.eq(input_ids, model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(hidden_states.device)
            embeddings = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        embeddings = embeddings.float().cpu().numpy()

        return dict(embeddings=embeddings)

    dataset = dataset.map(create_embeddings, num_proc=None, batched=True, batch_size=64, load_from_cache_file=False)
    embedding_columns_to_remove = [
        c for c in dataset["train"].column_names if c != "embeddings"
    ]
    if embedding_columns_to_remove:
        print(f"[add_embeddings] removing {embedding_columns_to_remove}", flush=True)
        dataset = dataset.remove_columns(embedding_columns_to_remove)
    dataset.save_to_disk(str(embedded_path))
    print(f"[add_embeddings] done, saved to {embedded_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("base_model")
    p.add_argument("uid")
    args = p.parse_args()
    print(f"[add_embeddings] start {args.dataset_dir} {args.base_model} uid={args.uid}", flush=True)
    run_add_embeddings(args.dataset_dir, args.base_model, args.uid)


if __name__ == "__main__":
    main()
