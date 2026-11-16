import statistics
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

sorted_relevant_metrics = ['C11', 'T1', 'C6', 'S5', 'C1', 'M2', 'M3', 'C8', 'C2', 'S4', 'C3', 'C4', 'C7', 'T2', 'S1', 'S2', 'C10', 'C12', 'C9', 'M1', 'T3', 'C5', 'S3']


def load_all(full_path, metrics, load_last=False):
    selection_predicate = min
    if load_last:
        selection_predicate = max
    results = []
    for path in full_path.iterdir():
        checkpoints = list(path.iterdir())
        selected_checkpoint = selection_predicate(checkpoints, key=lambda p: int(p.stem.split("checkpoint-")[-1]))
        # llm.add_predictions stores the logits only; the code metrics, idx and
        # target of the same rows are in the metrics dataset.
        logits = load_from_disk(selected_checkpoint)
        results.append(DatasetDict({
            split: concatenate_datasets([metrics[split], logits[split].select_columns(["logits"])], axis=1)
            for split in logits
        }))
        
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


def run_dependence_main() -> None:
    dataset_path = Path("./data/primevul-lookaround")
    model_paths = {
        "PDBert": dataset_path / "data__pdbert",
        "UniXcoder": dataset_path / "microsoft__unixcoder-base-nine",
        "LineVul": dataset_path / "microsoft__codebert-base",
        "StarCoder": dataset_path / "bigcode__starcoder2-7b",
        "CodeGen": dataset_path / "Salesforce__codegen25-7b-multi_P",
    }
    BIG_MODELS = ["StarCoder", "CodeGen"]

    all_results = {
        # manually checked: early stopping produced better checkpoint for all models
        # except big models, where the last checkpoint was always the best
        # this was verified for all runs
        # metrics of the functions after the round trip through that model's tokenizer
        name: load_all(path, load_from_disk(dataset_path / f"mcpp-{path.name}"), load_last=name in BIG_MODELS)
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
            diff_predictions = torch.as_tensor(list(diffed["train"]["diff_predictions"]) + list(diffed["val"]["diff_predictions"]) + list(diffed["test"]["diff_predictions"]))
            summed += torch.mean(torch.abs(diff_predictions)).item()
        print(name, summed / len(all_results[name]))


def main() -> None:
    run_dependence_main()


if __name__ == "__main__":
    main()
