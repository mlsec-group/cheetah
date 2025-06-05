import os
import math
import json
import numpy as np
import argparse

import torch

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification


relevant_metrics = ["C1", "C2", "C3", "C4", "V1", "V3", "V6", "V7", "V8", "V10", "x1", "x2", "x3", "x4", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "m1", "m2"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("base_model")
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
    model = AutoModelForSequenceClassification.from_pretrained(final_model_path, torch_dtype=torch.float16, device_map="auto")
    device = next(model.parameters()).device

    embedding_dataset = load_from_disk(os.path.join(final_model_path, "embedded"))
    embedding_dataset.set_format("numpy")
    model_suffix = "-"+args.base_model.replace("/", "__")
    features_dataset = load_from_disk(args.dataset_path + model_suffix)

    train_features = np.array([features_dataset["train"][k] for k in relevant_metrics]).T
    test_features = np.array([features_dataset["test"][k] for k in relevant_metrics]).T

    train_embeddings = np.concatenate([embedding_dataset["train"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["train"]))])
    test_embeddings = np.concatenate([embedding_dataset["test"][i]["embeddings"][np.newaxis, ...] for i in range(len(embedding_dataset["test"]))])
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