import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


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
        num_proc=1,
        input_columns=["input_ids", "attention_mask"],
        load_from_cache_file=False,
    )

    return logits_set


def process_checkpoint(checkpoint_path, tokenizer, dataset):
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    logits_set = process_model(model, tokenizer, dataset)

    return logits_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("base_model", default="microsoft/unixcoder-base-nine")
    parser.add_argument("uid", default=None)
    args = parser.parse_args()

    final_model_path = os.path.join("hf-model", args.base_model.replace("/", "__"))
    final_model_path = os.path.join(final_model_path, args.uid)
    
    if args.base_model == "Salesforce/codegen25-7b-multi_P":
        # specific revision to fix bug on main branch
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi_P", trust_remote_code=True, revision="c936d25f73e74984748a5ba39151b6bc7a056e2e")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.base_model in ["Salesforce/codegen25-7b-multi_P", "bigcode/starcoder2-7b"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_from_disk(args.dataset_path)
    model_max_length = 1024 if args.base_model == "microsoft/unixcoder-base-nine" else 512
    print(f"Tokenizing datasets with maximum length {model_max_length}")
    tokenize_with_tokenizer = lambda s: tokenize(s, tokenizer, model_max_length)
    dataset = dataset.map(tokenize_with_tokenizer, batched=True, num_proc=None, load_from_cache_file=False)

    for checkpoint in tqdm(os.listdir(final_model_path)):
        checkpoint_path = os.path.join(final_model_path, checkpoint)
        result = process_checkpoint(checkpoint_path, tokenizer, dataset)
        result_path = os.path.join(
            os.path.abspath(os.path.join(args.dataset_path, os.pardir)), 
            args.base_model.replace("/", "__"), 
            args.uid, 
            checkpoint
        )
        os.makedirs(result_path, exist_ok=True)
        result.save_to_disk(result_path)