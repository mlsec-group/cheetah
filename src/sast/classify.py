import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import f1_score, roc_curve, average_precision_score

NUM_STEPS = 100


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


def run_classify(data: Path) -> None:
    hf_path = data / "sast" / "hf"
    dataset = load_from_disk(hf_path)

    sast_tools = ["rats", "cppcheck", "semgrep"]

    for sast_tool in sast_tools:
        print(f"[*] Classify {sast_tool}")
        train_features = np.array([el[sast_tool] for el in dataset["train"]])
        test_features = np.array([el[sast_tool] for el in dataset["test"]])
        train_labels = np.array([el["labels"] for el in dataset["train"]])
        test_labels = np.array([el["labels"] for el in dataset["test"]])

        # Transform
        #train_features += 1
        #test_features += 1

        min_val = np.min(train_features)
        max_val = np.max(train_features)

        #train_features /= max_val
        #test_features /= max_val

        best_score = -1
        best_threshold = -1

        for threshold in tqdm(range(0, NUM_STEPS, 1), total=NUM_STEPS):
            score = f1_score(train_labels, train_features > threshold)

            if score > best_score:
                best_score = score
                best_threshold = threshold


        test_f1_score = f1_score(test_labels, test_features > best_threshold)
        test_average_precision_score = average_precision_score(test_labels, test_features > best_threshold)
        test_vds_score = calculate_vul_det_score(train_labels, train_features)

        print(f"f1: {test_f1_score}\nvds: {test_vds_score}\nauprc: {test_average_precision_score}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", type=Path, required=True)
    args = p.parse_args()
    run_classify(args.data)


if __name__ == "__main__":
    main()
