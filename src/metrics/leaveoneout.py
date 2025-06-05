import click
import pickle
import numpy as np
import autosklearn.metrics
from autosklearn.classification import AutoSklearnClassifier as Classifier
from pathlib import Path
from datasets import load_from_disk

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
)

from config import TOTAL_TIME, N_JOBS, BEST_MI as BEST
from util import calculate_vul_det_score, best_threshold


@click.command()
@click.option("--data-dir", required=True, type=click.Path(dir_okay=True, file_okay=False, exists=True, path_type=Path))
@click.option("--job-id", required=True, type=int)
def main(data_dir, job_id):
    job_id = job_id
    metric_no = job_id % 23

    relevant_metrics = [BEST[i] for i in range(len(BEST)) if i != metric_no]

    # Output directory
    output_dir = data_dir / "autosklearn-model-abo" / str(metric_no) / str(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    datasets = load_from_disk(data_dir)
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


if __name__ == "__main__":
    main()
