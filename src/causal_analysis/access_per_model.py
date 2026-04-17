import argparse
import os
import json
import numpy as np
from pathlib import Path

import torch

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification


relevant_metrics = ["C1", "C2", "C3", "C4", "V1", "V3", "V6", "V7", "V8", "V10", "x1", "x2", "x3", "x4", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "m1", "m2"]


def run_access_per_model(dataset_path: str, base_model: str, uid: str) -> None:
    final_model_path = os.path.join("hf-model", base_model.replace("/", "__"))
    final_model_path = os.path.join(final_model_path, uid)
    seeds = os.listdir(final_model_path)
    for seed in seeds:
        print("Searching for checkpoints in ", str(final_model_path/seed))
        checkpoints = list((Path(final_model_path)/seed).glob("checkpoint-*"))
        print("Checkpoints: ", str(checkpoints))
        assert len(checkpoints) > 0 and all(checkpoint.split("-")[1].isnumeric() for checkpoint in checkpoints)
        smallest_checkpoint = sorted(checkpoints, key=lambda s: int(s.split("-")[1]))[0]
        final_model_path = os.path.join(final_model_path, smallest_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            final_model_path, torch_dtype=torch.float16, device_map="auto"
        )

        embedding_dataset = load_from_disk(os.path.join(final_model_path, "embedded"))
        embedding_dataset.set_format("numpy")
        model_suffix = "-" + base_model.replace("/", "__")
        features_dataset = load_from_disk(dataset_path + model_suffix)

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
            real_feature = test_features[:, feature_index]
            feature_predictions.append(feature_prediction.tolist())
        feature_predictions = np.array(feature_predictions).T
        r2_scores = r2_score(test_features, feature_predictions, multioutput="raw_values").tolist()

        with open(os.path.join(final_model_path, "r2_results.json"), "w", encoding="utf8") as f:
            json.dump(dict(zip(relevant_metrics, r2_scores)), f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_path")
    p.add_argument("base_model")
    p.add_argument("uid")
    args = p.parse_args()
    run_access_per_model(args.dataset_path, args.base_model, args.uid)


if __name__ == "__main__":
    main()
