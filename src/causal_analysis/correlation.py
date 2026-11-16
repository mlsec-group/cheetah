import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from scipy.stats import pearsonr
from tqdm import tqdm

from paths import model_slug


def load_all_predictions(model_dir: Path, smallest_checkpoint: bool = True):
    checkpoints = []
    for uid in range(10):
        _checkpoints = list((model_dir/str(uid)).glob("checkpoint-*"))
        if len(_checkpoints) == 0:
            raise ValueError(f"No checkpoints found for {model_dir/str(uid)}")
        if smallest_checkpoint:
            _checkpoints = sorted(_checkpoints, key=lambda s: int(s.name.split("-")[1]))[0]
        else:
            _checkpoints = _checkpoints[-1]
        checkpoints.append(_checkpoints)
    predictions = []
    for checkpoint in checkpoints:
        with open(checkpoint / "predictions.json", "r", encoding="utf8") as f:
            predictions.append(np.asarray(json.load(f)))
    return predictions


def mean_correlation(predictions_one, predictions_two, same):
    correlations = []
    pb = range(len(predictions_one))
    if len(predictions_one) * len(predictions_two) > 1_000:
        pb = tqdm(pb)
    for i in pb:
        for j in range(len(predictions_two)):
            if same and i >= j:
                continue
            correlations.append(pearsonr(predictions_one[i], predictions_two[j]).statistic)
    return np.mean(correlations)

relevant_metrics = ["S1", "S2", "S3", "S4", "S5", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "M1", "M2", "M3", "T1", "T2", "T3"]


def run_correlation_main(dataset_path: Path) -> None:
    models = dataset_path / "hf-model"
    base_models = [
        "microsoft/unixcoder-base-nine",
        "data/pdbert",
        "microsoft/codebert-base",
        "bigcode/starcoder2-7b",
        "Salesforce/codegen25-7b-multi_P",
    ]
    predictions = {base_model: load_all_predictions(models / model_slug(base_model)) for base_model in base_models}

    # need to have consistent ordering
    max_namelen = max(len(m) for m in base_models)
    print("Correlation between model predictions:")
    print(" " * max_namelen + " " + " ".join(n[:4].ljust(4) for n in base_models))
    for i, base_model in enumerate(base_models):
        print(base_model.ljust(max_namelen), end="")
        data1 = predictions[base_model]
        for j in range(0, i+1):
            other = base_models[j]
            data2 = predictions[other]
            r = mean_correlation(data1, data2, i == j)
            print(f" {round(r, 2):0.2f}", end="")
        print("")

    metrics = {base_model: load_from_disk(dataset_path / f"mcpp-{model_slug(base_model)}")["test"] for base_model in base_models}
    labels = np.array(metrics[base_models[0]]["labels"])
    metrics = {base_model: np.array([metrics[base_model][k] for k in relevant_metrics]).T for base_model in base_models}

    correlations = {}
    for i, k in enumerate(relevant_metrics):
        rs = {}
        for base_model in base_models:
            rs[base_model] = mean_correlation(predictions[base_model], [metrics[base_model][:, i]], False)
            rs[f"{base_model}-true"] = mean_correlation([labels], [metrics[base_model][:, i]], False)
        correlations[k] = rs


    print("Correlation between metric values and model predictions")
    print("{" + ",".join(relevant_metrics) + "},")
    for base_model in base_models:
        print(base_model)
        print("{", end="")
        for ik, k in enumerate(relevant_metrics):
            if ik < len(relevant_metrics) - 1:
                print(f"{round(correlations[k][base_model], 4):.4f}, ", end="")
            else:
                print(f"{round(correlations[k][base_model], 4):.4f}", end="},")
        print("},")

    print("Correlation between metric values and true labels")
    print("{" + ",".join(relevant_metrics) + "},")
    for base_model in base_models:
        print(base_model)
        print("{", end="")
        for ik, k in enumerate(relevant_metrics):
            base_model_key = f"{base_model}-true"
            if ik < len(relevant_metrics) - 1:
                print(f"{round(correlations[k][base_model_key], 4):.4f}, ", end="")
            else:
                print(f"{round(correlations[k][base_model_key], 4):.4f}", end="},")
        print("},")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    args = parser.parse_args()
    run_correlation_main(args.dataset_path)


if __name__ == "__main__":
    main()
