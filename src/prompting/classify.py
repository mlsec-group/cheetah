import argparse
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_curve


def calculate_vul_det_score(ground_truth, predictions, target_fpr=0.005):
    # stolen from: https://github.com/DLVulDet/PrimeVul/blob/6f54687c84947b1d17486495440b37030d147289/calc_vd_score.py#L6C1-L43C33
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
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions, drop_intermediate=False)

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
    classified_preds = [1 if pred >= chosen_threshold else 0 for pred in predictions]

    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0

    return vds


def find_best_threshold(metric, labels, scores, steps=500):
    min_score = min(scores)
    max_score = max(scores)
    thresholds = np.linspace(min_score, max_score, steps)
    results = []
    for t in thresholds:
        try:
            results.append(metric(labels, scores > t))
        except:
            pass
    return thresholds[max(range(len(results)), key=results.__getitem__)]


def conf_interval(scores):
    confidence_interval_95 = 1.644850 * statistics.stdev(scores) / math.sqrt(len(scores))
    return confidence_interval_95


def calc_score(response):
    # Count method
    yes_count = len(re.findall("YES", response[-100:]))
    no_count = len(re.findall("NO", response[-100:]))
    return yes_count - no_count


def run_classify(dataset_dir: Path, model: str, size: str) -> None:
    prompting_dir = dataset_dir / "prompting"
    raw_dir = prompting_dir / "raw-batches"
    llm_dir = prompting_dir / "out"

    dataset = dataset_dir.name
    experiment = f"{model}_{dataset}_{size}"

    batch_inputs_fpath = raw_dir / f"{dataset}-raw-batch-{size}.jsonl"
    batch_results_fpaths = list(llm_dir.glob(f"{experiment}*"))

    input_data = [json.loads(line) for line in batch_inputs_fpath.read_text().split("\n")]
    label_map = {dp["hf-idx"]: dp["labels"] for dp in input_data}

    f1_scores = []
    average_precision_scores = []
    mcc_scores = []
    bacc_scores = []
    vds_scores = []

    for batch_fpath in batch_results_fpaths:
        print(batch_fpath)
        lines = batch_fpath.read_text().split("\n")
        data = [json.loads(line) for line in lines if len(line) > 2]

        responses = (dp["response"]["body"]["choices"][0]["message"]["content"] for dp in data)
        hf_ids = (dp["custom_id"].split("-")[-1] for dp in data)
        labels = np.array([label_map[int(hf_id)] for hf_id in hf_ids])
        scores = np.array([calc_score(response) for response in responses])

        best_threshold = find_best_threshold(f1_score, labels, scores)
        print(sum(scores > best_threshold) / len(scores))

        f1 = f1_score(labels, scores > best_threshold)
        average_precision = average_precision_score(labels, scores)
        mcc = matthews_corrcoef(labels, scores > best_threshold)
        bacc = balanced_accuracy_score(labels, scores > best_threshold)
        vds = calculate_vul_det_score(labels, scores)

        f1_scores.append(f1)
        average_precision_scores.append(average_precision)
        mcc_scores.append(mcc)
        bacc_scores.append(bacc)
        vds_scores.append(vds)

    f1_scores = np.array(f1_scores)
    average_precision_scores = np.array(average_precision_scores)

    print("f1 mean", np.mean(f1_scores))
    print("f1 conf-95", conf_interval(f1_scores))

    print("average precision mean", np.mean(average_precision_scores))
    print("average precision conf-95", conf_interval(average_precision_scores))

    print("mcc mean", np.mean(mcc_scores))
    print("mcc conf-95", conf_interval(mcc_scores))

    print("bacc mean", np.mean(bacc_scores))
    print("bacc conf-95", conf_interval(bacc_scores))

    print("vds mean", np.mean(vds_scores))
    print("vds conf-95", conf_interval(vds_scores))


def main() -> None:
    p = argparse.ArgumentParser(description="Score the downloaded responses of one model against the labels.")
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("model")
    p.add_argument("--size", default="s")
    args = p.parse_args()
    run_classify(args.dataset_dir, args.model, args.size)


if __name__ == "__main__":
    main()
