import os
import math
import argparse

import torch

from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, AutoTokenizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("base_model")
    parser.add_argument("uid", default=None)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    if args.uid is None:
        raise ValueError(f"invalid uid")
    final_model_path = os.path.join("hf-model", args.base_model.replace("/", "__"))
    final_model_path = os.path.join(final_model_path, args.uid)
    checkpoints = os.listdir(final_model_path)
    assert len(checkpoints) > 0 and all(checkpoint.split("-")[1].isnumeric() for checkpoint in checkpoints)
    smallest_checkpoint = sorted(checkpoints, key=lambda s: int(s.split("-")[1]))[0]
    final_model_path = os.path.join(final_model_path, smallest_checkpoint)

    if args.base_model == "Salesforce/codegen25-7b-multi_P":
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi_P", trust_remote_code=True, revision="c936d25f73e74984748a5ba39151b6bc7a056e2e")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.base_model in ["Salesforce/codegen25-7b-multi_P", "bigcode/starcoder2-7b"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForSequenceClassification.from_pretrained(final_model_path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    model_max_length = 1024 if args.base_model == "microsoft/unixcoder-base-nine" else 512
    print(f"Tokenizing datasets with maximum length {model_max_length}")
    def tokenize(sample):
        return tokenizer(sample["texts"], truncation=True, max_length=model_max_length)
    columns_to_remove = [c for c in dataset["train"].column_names if "label" not in c]
    print(f"Removing {columns_to_remove} after tokenization")
    dataset = dataset.map(tokenize, batched=True, num_proc=None, remove_columns=columns_to_remove, load_from_cache_file=False)

    print("Embedding datasets")
    device = next(model.parameters()).device
    collator = DataCollatorWithPadding(tokenizer, padding=True)
    @torch.no_grad()
    def create_embeddings(sample):
        inputs = collator(sample)
        inputs = inputs.to(device)

        hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        if args.base_model not in ["Salesforce/codegen25-7b-multi_P", "bigcode/starcoder2-7b"]:
            embeddings = hidden_states[-1][:, 0, :]
            embeddings = model.classifier.dense(embeddings)
        embeddings = embeddings.float().cpu().numpy()
        
        return dict(embeddings=embeddings)
    dataset = dataset.map(create_embeddings, num_proc=None, batched=True, batch_size=64, load_from_cache_file=False)
    
    dataset.save_to_disk(os.path.join(final_model_path, "embedded"))