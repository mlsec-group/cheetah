import argparse
import json
import os

import numpy as np
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from paths import model_slug

relevant_metrics = ["S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "M1", "M2", "M3", "T1", "T2", "T3"]


def run_cross_information(dataset_path: str, base_model: str, uid: str) -> None:
    slug = model_slug(base_model)
    final_model_path = os.path.join("hf-model", slug)
    final_model_path = os.path.join(final_model_path, uid)
    checkpoints = os.listdir(final_model_path)
    assert len(checkpoints) > 0 and all(checkpoint.split("-")[1].isnumeric() for checkpoint in checkpoints)
    smallest_checkpoint = sorted(checkpoints, key=lambda s: int(s.split("-")[1]))[0]
    final_model_path = os.path.join(final_model_path, smallest_checkpoint)

    embedding_dataset = load_from_disk(os.path.join(final_model_path, "embedded"))
    embedding_dataset.set_format("numpy")
    model_suffix = "-" + slug
    features_dataset = load_from_disk(dataset_path + model_suffix)

    train_features = np.array([features_dataset["train"][k] for k in relevant_metrics]).T
    test_features = np.array([features_dataset["test"][k] for k in relevant_metrics]).T
    train_labels = np.array(features_dataset["train"]["labels"])
    test_labels = np.array(features_dataset["test"]["labels"])

    train_embeddings = np.concatenate(
        [embedding_dataset["train"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["train"]))]
    )
    test_embeddings = np.concatenate(
        [embedding_dataset["test"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["test"]))]
    )

    train_combined = np.hstack([train_embeddings, train_features])
    test_combined = np.hstack([test_embeddings, test_features])

    scaler = StandardScaler().fit(train_embeddings)
    model = LogisticRegression(class_weight="balanced").fit(scaler.transform(train_embeddings), train_labels)
    f1_normal = f1_score(test_labels, model.predict(scaler.transform(test_embeddings)))

    scaler = StandardScaler().fit(train_combined)
    model = LogisticRegression(class_weight="balanced").fit(scaler.transform(train_combined), train_labels)
    f1_combined = f1_score(test_labels, model.predict(scaler.transform(test_combined)))

    with open(os.path.join(final_model_path, "concatenation_results.json"), "w", encoding="utf8") as f:
        json.dump([f1_normal, f1_combined], f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_path")
    p.add_argument("base_model")
    p.add_argument("uid")
    args = p.parse_args()
    run_cross_information(args.dataset_path, args.base_model, args.uid)


if __name__ == "__main__":
    main()
