import argparse
import pickle
import numpy as np
import json
import autosklearn.metrics
from autosklearn.classification import AutoSklearnClassifier as Classifier
from pathlib import Path
from datasets import load_from_disk

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
)

from metrics.config import TOTAL_TIME, N_JOBS, BEST_MI as BEST
from metrics.util import calculate_vul_det_score, best_threshold


def run_isolation(data_dir: Path, job_id: int) -> None:
    metric_no = job_id % len(BEST)

    # Output directory
    output_dir = data_dir / "autosklearn-model-isolation" / str(metric_no) / str(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    datasets = load_from_disk(data_dir)
    relevant_metrics = [BEST[metric_no]]

    train_features = np.array([datasets["train"][k] for k in relevant_metrics]).T
    val_features = np.array([datasets["val"][k] for k in relevant_metrics]).T
    test_features = np.array([datasets["test"][k] for k in relevant_metrics]).T

    # Auto-Sklearn doesn't work when there is too low variance in the data, so we add random features
    num_missing = len(BEST) - 1
    rand_train = np.random.rand(train_features.shape[0], num_missing)
    train_features = np.concatenate([train_features, rand_train], axis=1)
    rand_val = np.random.rand(val_features.shape[0], num_missing)
    val_features = np.concatenate([val_features, rand_val], axis=1)
    rand_test = np.random.rand(test_features.shape[0], num_missing)
    test_features = np.concatenate([test_features, rand_test], axis=1)

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
    run_isolation(args.data_dir, args.job_id)


if __name__ == "__main__":
    main()
