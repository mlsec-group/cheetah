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


def process_model(model, tokenizer, dataset):
    device = next(model.parameters()).device
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    @torch.no_grad()
    def create_logits(input_ids, attention_mask):
        inputs = collator(dict(input_ids=input_ids, attention_mask=attention_mask))
        inputs = inputs.to(device)

        logits = model(**inputs).logits

        return dict(logits=logits.cpu().flatten().tolist())

    logits_set = dataset.map(
        create_logits,
        batched=True,
        batch_size=8,
        num_proc=None,
        input_columns=["input_ids", "attention_mask"],
        load_from_cache_file=False,
    )

    return logits_set


def process_checkpoint(checkpoint_path, tokenizer, dataset):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return process_model(model, tokenizer, dataset)


def run_add_predictions(
    dataset_dir: Path,
    base_model: str,
    uid: str,
    models_dir: Path | None = None,
) -> None:
    dataset_dir = dataset_dir.resolve()
    models_dir = (models_dir or dataset_dir).resolve()
    slug = model_slug(base_model)
    checkpoints_dir = models_dir / "hf-model" / slug / str(uid)
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints directory: {checkpoints_dir}")

    checkpoints = [
        p for p in checkpoints_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-") and p.name.split("-")[1].isnumeric()
    ]
    assert checkpoints, f"No checkpoints under {checkpoints_dir}"
    checkpoint = min(checkpoints, key=lambda p: int(p.name.split("-")[1]))

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

    hf_path = dataset_dir / "hf"
    if not hf_path.is_dir():
        raise FileNotFoundError(
            f"HF dataset not found at {hf_path}. "
            f"Pass the dataset root (e.g. data/primevul-lookaround), not a relative name."
        )
    dataset = load_from_disk(str(hf_path))
    model_max_length = 1024 if base_model == "microsoft/unixcoder-base-nine" else 512
    print(
        f"[add_predictions] dataset={dataset_dir} models={models_dir} uid={uid} "
        f"checkpoint={checkpoint.name} max_length={model_max_length}",
        flush=True,
    )
    columns_to_remove = [c for c in dataset["train"].column_names if "label" not in c]
    print(f"[add_predictions] tokenize map, removing {columns_to_remove}", flush=True)
    dataset = dataset.map(
        tokenize, batched=True, num_proc=None,
        remove_columns=columns_to_remove,
        load_from_cache_file=False,
        fn_kwargs=dict(tokenizer=tokenizer, truncation_length=model_max_length),
    )

    result_path = dataset_dir / slug / str(uid) / checkpoint.name
    if (result_path / "dataset_dict.json").exists():
        print(f"[add_predictions] {result_path} already exists, skipping", flush=True)
        return

    print(f"[add_predictions] inference map for {slug}/{uid}/{checkpoint.name}", flush=True)
    result = process_checkpoint(checkpoint, tokenizer, dataset)
    result.save_to_disk(str(result_path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_dir", type=Path, help="Dataset root (loads hf/, writes prediction outputs)")
    p.add_argument("base_model")
    p.add_argument("uid")
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Dataset root containing hf-model/ (default: same as dataset_dir)",
    )
    args = p.parse_args()
    print(f"[add_predictions] start {args.dataset_dir} {args.base_model} uid={args.uid}", flush=True)
    run_add_predictions(args.dataset_dir, args.base_model, args.uid, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
