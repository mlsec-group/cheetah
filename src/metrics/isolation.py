import os
import argparse
import pickle
import math
import numpy as np
import autosklearn.metrics
from autosklearn.classification import AutoSklearnClassifier as Classifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import normalized_mutual_info_score
from pathlib import Path
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score, roc_curve, log_loss, brier_score_loss
)


TOTAL_TIME = 10*60
N_JOBS = 4

# Like in MI matrix
BEST = ["T2", "C7", "M2", "M3", "C8", "C10", "C11", "C3", "C2", "C4", "S5", "C1", "T1", "C6", "S4", "S1", "C9", "M1", "S2", "C12", "T3", "S3", "C5"]


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


def best_threshold(metric, labels, scores):
    thresholds = np.linspace(0, 1, 500)
    results = []
    for t in thresholds:
        try:
            results.append(metric(labels, scores > t))
        except:
            pass
    return thresholds[max(range(len(results)), key=results.__getitem__)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("job_id")
    args = parser.parse_args()

    job_id = int(args.job_id)
    metric_no = (job_id - 1) % 23
    repitition = (job_id - 1) // 23

    # Output directory
    output_dir = Path(args.data_dir).parent / "autosklearn-model-sf" / str(metric_no) / str(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    datasets = load_from_disk(args.data_dir)
    relevant_metrics = [BEST[metric_no]]

    train_features = np.array([datasets["train"][k] for k in relevant_metrics]).T
    val_features = np.array([datasets["val"][k] for k in relevant_metrics]).T
    test_features = np.array([datasets["test"][k] for k in relevant_metrics]).T

    train_labels = np.array(datasets["train"]["labels"])
    val_labels = np.array(datasets["val"]["labels"])
    test_labels = np.array(datasets["test"]["labels"])

    # Scale features
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)
    
    # non exhaustive cross validation
    selected_indices = np.random.choice(np.arange(len(train_scaled)), int(len(train_scaled) * 0.9))
    train_scaled = train_scaled[selected_indices]
    train_labels = train_labels[selected_indices]

    # AutoML
    automl = Classifier(
        time_left_for_this_task=TOTAL_TIME,
        n_jobs=N_JOBS,
        memory_limit=8192,
        metric=autosklearn.metrics.f1,
    )
    automl = automl.fit(train_scaled, train_labels)

    print(automl.leaderboard())
    prediction_scores = automl.predict_proba(val_scaled)[:, 1]
    threshold = best_threshold(f1_score, val_labels, prediction_scores)
    prediction_scores = automl.predict_proba(test_scaled)[:, 1]
    predictions = prediction_scores > threshold
    metrics = {
        "AUC": roc_auc_score(test_labels, prediction_scores),
        "AP": average_precision_score(test_labels, prediction_scores),
        "VD-S": calculate_vul_det_score(test_labels, prediction_scores),
        "MCC": matthews_corrcoef(test_labels, predictions),
        "BAcc": balanced_accuracy_score(test_labels, predictions),
        "F1": f1_score(test_labels, predictions),
        "Prec": precision_score(test_labels, predictions),
        "Rec": recall_score(test_labels, predictions),
    }

    print(f"RESULT-METRICS {' '.join(BEST)}")
    for metric, score in metrics.items():
        print(f"RESULT-{metric} {metric_no} {score}")

    # Save model
    model_fpath = output_dir / "trained_model.pkl"
    model_fpath.write_bytes(pickle.dumps(automl))

