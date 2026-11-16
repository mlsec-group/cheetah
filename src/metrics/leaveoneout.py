import argparse
import json
import pickle
from pathlib import Path

import autosklearn.metrics
import numpy as np
from autosklearn.classification import AutoSklearnClassifier as Classifier
from datasets import load_from_disk
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from metrics.config import BEST_MI as BEST
from metrics.config import N_JOBS, TOTAL_TIME
from metrics.util import best_threshold, calculate_vul_det_score


def run_leaveoneout(data_dir: Path, job_id: int) -> None:
    metric_no = job_id % len(BEST)

    relevant_metrics = [BEST[i] for i in range(len(BEST)) if i != metric_no]

    # Output directory
    output_dir = data_dir / "autosklearn-model-leaveoneout" / str(metric_no) / str(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    datasets = load_from_disk(str(data_dir))
    train_features = np.array([datasets["train"][m] for m in relevant_metrics]).T
    val_features = np.array([datasets["val"][m] for m in relevant_metrics]).T
    test_features = np.array([datasets["test"][m] for m in relevant_metrics]).T

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

    # Save predictions
    predictions_fpath = output_dir / "predictions.json"
    predictions_fpath.write_text(json.dumps(predictions.tolist(), indent=2))

    # Save results
    result_fpath = output_dir / "results.json"
    result_fpath.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--job-id", type=int, required=True)
    args = p.parse_args()
    run_leaveoneout(args.data_dir, args.job_id)


if __name__ == "__main__":
    main()
