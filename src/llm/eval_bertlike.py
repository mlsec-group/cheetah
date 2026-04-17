import argparse
import json
import math

import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoConfig
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, average_precision_score, roc_auc_score, roc_curve
)


def calculate_vul_det_score(ground_truth, predictions, target_fpr=0.005):
    # inspired by: https://github.com/DLVulDet/PrimeVul/blob/6f54687c84947b1d17486495440b37030d147289/calc_vd_score.py#L6C1-L43C33
    """
    Calculate the vulnerability detection score (VD-S) given a tolerable FPR.

    Args:
    - predictions: List of model prediction probabilities for the positive class.
    - ground_truth: List of ground truth labels, where 1 means vulnerable class, and 0 means benign class.
    - target_fpr: The tolerable false positive rate.

    Returns:
    - vds: Calculated vulnerability detection score given the acceptable .
    - threshold: The classification threashold for vulnerable prediction.
    """

    # Calculate FPR, TPR, and thresholds using ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions, drop_intermediate=True)

    # Filter thresholds where FPR is less than or equal to the target FPR
    valid_indices = np.where(fpr <= target_fpr)[0]

    # Choose the threshold with the largest FPR that is still below the target FPR, if possible
    if len(valid_indices) > 0:
        idx = valid_indices[-1]  # Last index where FPR is below or equal to target FPR
    else:
        # If no such threshold exists (unlikely), default to the closest to the target FPR
        idx = np.abs(fpr - target_fpr).argmin()

    chosen_threshold = thresholds[idx]

    # Classify predictions based on the chosen threshold
    classified_preds = [1 if pred.item() >= chosen_threshold else 0 for pred in predictions]

    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0

    return vds


def calc_metrics(prediction_scores, labels, do_round=True):
    cpu_labels = labels
    if torch.is_tensor(labels):
        cpu_labels = labels.detach().cpu().numpy()
    cpu_labels = np.asarray(cpu_labels).reshape(-1)
    if torch.is_tensor(prediction_scores):
        prediction_scores = prediction_scores.detach().cpu().numpy()
    predictions = (prediction_scores > 0.5)
    metrics = {
        "AUC": roc_auc_score(cpu_labels, prediction_scores),
        "AP": average_precision_score(cpu_labels, prediction_scores),
        "VD-S": calculate_vul_det_score(cpu_labels, prediction_scores),
        "MCC": matthews_corrcoef(cpu_labels, predictions),
        "BAcc": balanced_accuracy_score(cpu_labels, predictions),
        "F1": f1_score(cpu_labels, predictions),
        "Prec": precision_score(cpu_labels, predictions),
        "Rec": recall_score(cpu_labels, predictions),
    }

    if do_round:
        metrics = {
            k: round(float(v), 4)
            for k, v in metrics.items()
        }

    return metrics


def evaluate_model(model, tokenizer, val_set, do_round=False):
    device = next(model.parameters()).device
    collator = DataCollatorWithPadding(tokenizer, padding=True)
    @torch.no_grad()
    def create_logits(sample):
        inputs = collator(sample)
        inputs = inputs.to(device)

        logits = model(**inputs).logits

        return dict(logits=logits.cpu().flatten().tolist())

    logits = val_set.map(create_logits, batched=True, batch_size=64, num_proc=None, load_from_cache_file=False)

    logits = torch.as_tensor(logits["logits"])
    predictions = torch.sigmoid(logits)

    return calc_metrics(predictions, val_set["labels"], do_round=do_round), predictions.flatten().cpu().tolist()


def evaluate_checkpoint(checkpoint_path, tokenizer, val_set):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, torch_dtype=torch.float16, device_map="auto")
    metrics, predictions = evaluate_model(model, tokenizer, val_set)

    with open(checkpoint_path / "results.json", "w", encoding="utf8") as f:
        json.dump(metrics, f)
    with open(checkpoint_path / "predictions.json", "w", encoding="utf8") as f:
        json.dump(predictions, f)


def tokenize(sample, tokenizer, truncation_length):
    return tokenizer(sample["texts"], truncation=True, max_length=truncation_length)


def collect_checkpoints(path):
    if not path.is_dir():
        return []

    ret = []
    if all((path / expected_file).exists() for expected_file in ["model.safetensors", "config.json"]):
        if (path / "results.json").exists() and (path / "predictions.json").exists():
            pass
        ret.append(path)


    for p in path.iterdir():
        ret.extend(collect_checkpoints(p))

    return ret


def run_eval_bertlike(models_path: Path, dataset_dir: str, base_model: str = "microsoft/unixcoder-base-nine") -> None:
    resolved = models_path / base_model.replace("/", "__")
    if not resolved.is_dir():
        raise ValueError("Provided path should be a directory containing all checkpoints to be tested in respective folders")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model_config = AutoConfig.from_pretrained(base_model)
    model_max_length = min(
        tokenizer.model_max_length,
        2**int(math.log(model_config.max_position_embeddings, 2)), # biggest possible power of two that is smaller
    )
    print(f"Tokenizing datasets with maximum length {model_max_length}")
    tokenize_with_tokenizer = lambda s: tokenize(s, tokenizer, model_max_length)
    test_set = load_from_disk(dataset_dir)["test"]
    columns_to_remove = [c for c in test_set.column_names if "label" not in c]
    print(f"Removing {columns_to_remove} after tokenization")
    test_set = test_set.map(
        tokenize_with_tokenizer, batched=True, num_proc=None, remove_columns=columns_to_remove, load_from_cache_file=False
    )

    checkpoints = collect_checkpoints(resolved)

    for checkpoint in tqdm(checkpoints):
        evaluate_checkpoint(checkpoint, tokenizer, test_set)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models_path", type=Path)
    p.add_argument("dataset_dir")
    p.add_argument("base_model", nargs="?", default="microsoft/unixcoder-base-nine")
    args = p.parse_args()
    run_eval_bertlike(args.models_path, args.dataset_dir, args.base_model)


if __name__ == "__main__":
    main()