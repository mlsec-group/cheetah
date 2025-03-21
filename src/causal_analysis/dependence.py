import torch
import statistics

import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_from_disk

from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


MAPPING = {
    "C1": "C1",
    "C2": "C2",
    "C3": "C3",
    "C4": "C4",
    "V1": "C5",
    "V3": "M3",
    "V6": "C6",
    "V7": "C7",
    "V8": "C8",
    "V10": "S5",
    "x1":  "C9",
    "x2": "C10",
    "x3": "C11",
    "x4": "C12",
    "t1": "T1",
    "t2": "T2",
    "t3": "T3",
    "s1": "S1",
    "s2": "S2",
    "s3": "S3",
    "s4": "S4",
    "m1": "M1",
    "m2": "M2",
}
INVERSE_MAPPING = {v: k for k,v in MAPPING.items()}

sorted_relevant_metrics = ['C11', 'T1', 'C6', 'S5', 'C1', 'M2', 'M3', 'C8', 'C2', 'S4', 'C3', 'C4', 'C7', 'T2', 'S1', 'S2', 'C10', 'C12', 'C9', 'M1', 'T3', 'C5', 'S3']
sorted_relevant_metrics = [INVERSE_MAPPING[k] for k in sorted_relevant_metrics]


def load_all(full_path, load_last=False):
    selection_predicate = min
    if load_last:
        selection_predicate = max
    results = []
    for path in full_path.iterdir():
        checkpoints = list(path.iterdir())
        selected_checkpoint = selection_predicate(checkpoints, key=lambda p: int(p.stem.split("checkpoint-")[-1]))
        results.append(load_from_disk(selected_checkpoint))
        
    return results


def add_cyclic_diffs(dataset):
    dataset = dataset.sort("idx")
    def process_idx(sample):
        assert all(sample["idx"][i] == sample["idx"][0] for i in range(4))
        index_past_version = min(i for i in range(4) if sample["target"][i] == 2)
        index_vuln_version = min(i for i in range(4) if sample["target"][i] == 1)
        index_benign_version = min(i for i in range(4) if sample["target"][i] == 3)
        index_future_version = min(i for i in range(4) if sample["target"][i] == 4)
        cyclic_next_index = {
            index_past_version: index_vuln_version,
            index_vuln_version: index_benign_version,
            index_benign_version: index_future_version,
            index_future_version: index_past_version,
        }
        sample["predictions"] = torch.sigmoid(torch.as_tensor(sample["logits"])).tolist()
        results = {}
        for column_name in sorted_relevant_metrics + ["logits", "predictions"]:
            ret = []
            column = sample[column_name]
            for i in range(4):
                compare_to = cyclic_next_index[i]
                ret.append(column[compare_to] - column[i])
            results[f"diff_{column_name}"] = ret
        return results
    return dataset.map(process_idx, batch_size=4, batched=True, load_from_cache_file=False)


def to_regression(dataset):
    X = []
    y = []
    for sample in dataset:
        total_diff = sum(abs(sample[f"diff_{k}"]) for k in sorted_relevant_metrics)
        if total_diff < 1e-5:
            continue
        y.append(sample["diff_predictions"])
        X.append([sample[f"diff_{k}"] for k in sorted_relevant_metrics])
    return X, y


def format_result(vals):
    mean = statistics.mean(vals)
    ci = 1.64 * statistics.stdev(vals) / len(vals)**0.5
    def format_number(number):
        number = f"{round(number, 4):.4f}"
        return str(number).rjust(7)
    return f"{format_number(mean)} ± {format_number(ci)}"


if __name__ == "__main__":
    dataset_path = Path("./data/primevul-lookaround")
    model_paths = {
        "PDBert": dataset_path / "data__pdbert",
        "UniXcoder": dataset_path / "microsoft__unixcoder-base-nine",
        "LineVul": dataset_path / "microsoft__codebert-base",
        "StarCoder": dataset_path / "Salesforce__codegen25-7b-multi_P",
        "CodeGen": dataset_path / "Salesforce__codegen25-7b-multi_P",
    }
    BIG_MODELS = ["StarCoder", "CodeGen"]

    all_results = {
        name: load_all(path, load_last=name in BIG_MODELS)
        for name, path in model_paths.items()
    }

    all_diffs = {
        name: [add_cyclic_diffs(sets) for sets in all_sets]
        for name, all_sets in all_results.items()
    }

    all_regression = {
        name: [
            {
                "train": to_regression(sets["train"]),
                "val": to_regression(sets["val"]),
                "test": to_regression(sets["test"]),
            } for sets in tqdm(all_sets)
        ]
        for name, all_sets in all_diffs.items()
    }

    model_names = list(all_results.keys())
    train_r2s = {name: [] for name in model_names}
    val_r2s = {name: [] for name in model_names}
    test_r2s = {name: [] for name in model_names}
    for name in tqdm(model_names):
        for sets in tqdm(all_regression[name], leave=False):
            X_train, y_train = sets["train"]
            X_val,   y_val   = sets["val"]
            X_test,  y_test  = sets["test"]

            X_train = np.abs(X_train)
            X_val   = np.abs(X_val)
            X_test  = np.abs(X_test)

            y_train = np.log1p(np.abs(y_train))
            y_val   = np.log1p(np.abs(y_val))
            y_test  = np.log1p(np.abs(y_test))
            
            regressor = make_pipeline(
                StandardScaler(),
                LinearRegression(),
            )
            regressor = regressor.fit(X_train, y_train)
            
            train_r2s[name].append(r2_score(y_train, regressor.predict(X_train)))
            val_r2s[name].append(r2_score(y_val, regressor.predict(X_val)))
            test_r2s[name].append(r2_score(y_test, regressor.predict(X_test)))


    for name in model_names:
        print(name)
        print(" ", "Train", format_result(train_r2s[name]))
        print(" ", "Val  ", format_result(val_r2s[name]))
        print(" ", "Test ", format_result(test_r2s[name]))


    for name in model_names:
        summed = 0
        for diffed in all_diffs[name]:
            diff_predictions = torch.as_tensor(diffed["train"]["diff_predictions"] + diffed["val"]["diff_predictions"] + diffed["test"]["diff_predictions"])
            summed += torch.mean(torch.abs(diff_predictions)).item()
        print(name, summed/len(all_results[name]))
