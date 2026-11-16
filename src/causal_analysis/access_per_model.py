import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

from paths import model_slug

relevant_metrics = ["S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "M1", "M2", "M3", "T1", "T2", "T3"]


def run_access_per_model(dataset_path: Path, base_model: str, uid: int) -> None:
    slug = model_slug(base_model)
    model_dir = dataset_path / "hf-model" / slug
    print("Searching for checkpoints in ", str(model_dir/(str(uid))))
    checkpoints = list((model_dir/(str(uid))).glob("checkpoint-*"))
    checkpoints = [c for c in checkpoints if (c/"embedded").is_dir()]
    print("Checkpoints: ", str(checkpoints))
    assert len(checkpoints) > 0 and all(checkpoint.name.split("-")[1].isnumeric() for checkpoint in checkpoints)
    smallest_checkpoint = sorted(checkpoints, key=lambda s: int(s.name.split("-")[1]))[0]
    final_model_path = smallest_checkpoint
    results_path = final_model_path / "r2_results.json"
    if results_path.exists():
        print(f"[+] Skipping checkpoint: {final_model_path} because results already exist")
        return
    print(f"[+] Processing checkpoint: {final_model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        final_model_path, torch_dtype=torch.float16, device_map="auto"
    )

    embedding_dataset = load_from_disk(final_model_path / "embedded")
    embedding_dataset.set_format("numpy")
    features_dataset = load_from_disk(dataset_path / f"mcpp-{slug}")

    train_features = np.array([features_dataset["train"][k] for k in relevant_metrics]).T
    test_features = np.array([features_dataset["test"][k] for k in relevant_metrics]).T

    train_embeddings = np.concatenate(
        [embedding_dataset["train"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["train"]))]
    )
    test_embeddings = np.concatenate(
        [embedding_dataset["test"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["test"]))]
    )
    if hasattr(model, "classifier"):
        train_embeddings = np.tanh(train_embeddings)
        test_embeddings = np.tanh(test_embeddings)

    train_log_features = np.log1p(train_features)
    linear_models = []
    for feature_index in tqdm(range(train_log_features.shape[1])):
        linear_model = LinearRegression().fit(train_embeddings, train_log_features[:, feature_index])
        linear_models.append(linear_model)

    feature_predictions = []
    for feature_index, linear_model in enumerate(linear_models):
        feature_prediction = np.expm1(linear_model.predict(test_embeddings))
        feature_predictions.append(feature_prediction.tolist())
    feature_predictions = np.array(feature_predictions).T
    r2_scores = r2_score(test_features, feature_predictions, multioutput="raw_values").tolist()

    results = dict(zip(relevant_metrics, r2_scores))
    with open(results_path, "w", encoding="utf8") as f:
        json.dump(results, f)
    print(f"[+] finished. R2 scores: \n{json.dumps(results, indent=4)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_path", type=Path)
    p.add_argument("base_model")
    p.add_argument("uid", type=int)
    args = p.parse_args()
    run_access_per_model(args.dataset_path, args.base_model, args.uid)


if __name__ == "__main__":
    main()
