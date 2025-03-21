import os
import json
import numpy as np
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from datasets import load_from_disk


relevant_metrics = ["C1", "C2", "C3", "C4", "V1", "V3", "V6", "V7", "V8", "V10", "x1", "x2", "x3", "x4", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "m1", "m2"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("base_model", default="microsoft/unixcoder-base-nine")
    parser.add_argument("uid", default=None)
    args = parser.parse_args()

    if args.uid is None:
        raise ValueError(f"invalid uid")
    final_model_path = os.path.join("hf-model", args.base_model.replace("/", "__"))
    final_model_path = os.path.join(final_model_path, args.uid)
    checkpoints = os.listdir(final_model_path)
    assert len(checkpoints) > 0 and all(checkpoint.split("-")[1].isnumeric() for checkpoint in checkpoints)
    smallest_checkpoint = sorted(checkpoints, key=lambda s: int(s.split("-")[1]))[0]
    final_model_path = os.path.join(final_model_path, smallest_checkpoint)

    embedding_dataset = load_from_disk(os.path.join(final_model_path, "embedded"))
    embedding_dataset.set_format("numpy")
    model_suffix = "-"+args.base_model.replace("/", "__")
    features_dataset = load_from_disk(args.dataset_path + model_suffix)

    train_features = np.array([features_dataset["train"][k] for k in relevant_metrics]).T
    val_features = np.array([features_dataset["val"][k] for k in relevant_metrics]).T
    test_features = np.array([features_dataset["test"][k] for k in relevant_metrics]).T
    train_labels = np.array(features_dataset["train"]["labels"])
    val_labels = np.array(features_dataset["val"]["labels"])
    test_labels = np.array(features_dataset["test"]["labels"])

    train_embeddings = np.concatenate([embedding_dataset["train"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["train"]))])
    val_embeddings = np.concatenate([embedding_dataset["val"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["val"]))])
    test_embeddings = np.concatenate([embedding_dataset["test"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["test"]))])

    train_combined = np.hstack([train_embeddings, train_features])
    val_combined = np.hstack([val_embeddings, val_features])
    test_combined = np.hstack([test_embeddings, test_features])

    scaler = StandardScaler().fit(train_embeddings)
    model = LogisticRegression(class_weight="balanced").fit(scaler.transform(train_embeddings), train_labels)
    f1_normal = f1_score(test_labels, model.predict(scaler.transform(test_embeddings)))

    scaler = StandardScaler().fit(train_combined)
    model = LogisticRegression(class_weight="balanced").fit(scaler.transform(train_combined), train_labels)
    f1_combined = f1_score(test_labels, model.predict(scaler.transform(test_combined)))

    with open(os.path.join(final_model_path, "concatenation_results.json"), "w", encoding="utf8") as f:
        json.dump([f1_normal, f1_combined], f)