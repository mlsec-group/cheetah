import json
import numpy as np

from pathlib import Path

from tqdm import tqdm
from scipy.stats import pearsonr
from datasets import load_from_disk

def load_all_predictions(path: Path):
    paths = path.rglob("predictions.json")
    predictions = []
    for path in paths:
        if "checkpoint" in str(path.parent):
            # check if smallest checkpoint
            model_root = path.parent.parent
            smallest_path = min((p for p in model_root.iterdir()), key=lambda p: int(str(p).split("checkpoint-")[-1]))
            if path.parent != smallest_path:
                continue
        with open(path, "r", encoding="utf8") as f:
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

def run_correlation_main() -> None:
    data = Path("./data/primevul")
    models = Path(".") / "hf-model"
    unixcoder_predictions = load_all_predictions(models / "microsoft__unixcoder-base-nine")
    pdbert_predictions = load_all_predictions(models / "data__pdbert")
    codebert_predictions = load_all_predictions(models / "microsoft__codebert-base")
    starcoder_predictions = load_all_predictions(models / "bigcode__starcoder2-7b")
    codegen_predictions = load_all_predictions(models / "Salesforce__codegen25-7b-multi_P")
    scm_predictions = load_all_predictions(Path(".") / "sklearn-model/autosklearn-model-10min")

    all_predictions = {
        "unixcoder": unixcoder_predictions[:10],
        "pdbert": pdbert_predictions[:10],
        "codebert": codebert_predictions[:10],
        "scm": scm_predictions[:10],
        "codegen": codegen_predictions,
        "starcoder": starcoder_predictions,
    }
    
    # need to have consistent ordering
    names = list(all_predictions.keys())
    longest_name = max(len(n) for n in names)
    print("Correlation between model predictions:")
    print(" " * longest_name + " " + " ".join(n[:4].ljust(4) for n in names))
    for i, row_name in enumerate(names):
        print(row_name.ljust(longest_name), end="")
        data1 = all_predictions[row_name]
        for j in range(0, i+1):
            col_name = names[j]
            data2 = all_predictions[col_name]
            r = mean_correlation(data1, data2, i == j)
            print(f" {round(r, 2):0.2f}", end="")
        print("")
    
    unixcoder_metrics = load_from_disk(data / "mcpp-microsoft__unixcoder-base-nine")["test"]
    pdbert_metrics = load_from_disk(data / "mcpp-microsoft__codebert-base")["test"]
    codebert_metrics = load_from_disk(data / "mcpp-microsoft__codebert-base")["test"]
    starcoder_metrics = load_from_disk(data / "mcpp-microsoft__codebert-base")["test"]
    codegen_metrics = load_from_disk(data / "mcpp-microsoft__codebert-base")["test"]
    real_metrics = load_from_disk(data / "mcpp")["test"]


    relevant_metrics = ["C1", "C2", "C3", "C4", "V1", "V3", "V6", "V7", "V8", "V10", "x1", "x2", "x3", "x4", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "m1", "m2"]
    labels = np.array(unixcoder_metrics["labels"])
    unixcoder_metrics = np.array([unixcoder_metrics[k] for k in relevant_metrics]).T
    pdbert_metrics = np.array([pdbert_metrics[k] for k in relevant_metrics]).T
    codebert_metrics = np.array([codebert_metrics[k] for k in relevant_metrics]).T
    codegen_metrics = np.array([codegen_metrics[k] for k in relevant_metrics]).T
    starcoder_metrics = np.array([starcoder_metrics[k] for k in relevant_metrics]).T
    real_metrics = np.array([real_metrics[k] for k in relevant_metrics]).T


    all_metrics = {
        "unixcoder": unixcoder_metrics,
        "pdbert": pdbert_metrics,
        "codebert": codebert_metrics,
        "codegen": codegen_metrics,
        "starcoder": starcoder_metrics,
    }
    correlations = {}
    for i, k in enumerate(relevant_metrics):
        rs = {}
        for model_name in all_metrics.keys():
            r = mean_correlation(all_predictions[model_name], [all_metrics[model_name][:, i]], False)
            rs[model_name] = r
        r = mean_correlation([labels], [real_metrics[:, i]], False)
        rs["label"] = r
        correlations[k] = rs

    sorted_relevant_metrics = ['C11', 'T1', 'C6', 'S5', 'C1', 'M2', 'M3', 'C8', 'C2', 'S4', 'C3', 'C4', 'C7', 'T2', 'S1', 'S2', 'C10', 'C12', 'C9', 'M1', 'T3', 'C5', 'S3']

    print("Correlation between metric values and model predictions")
    print("{" + ",".join(sorted_relevant_metrics) + "},")
    for model_name in correlations[INVERSE_MAPPING[sorted_relevant_metrics[0]]].keys():
        print(model_name)
        print("{", end="")
        for ik, k in enumerate(sorted_relevant_metrics):
            k = INVERSE_MAPPING[k]
            if ik < len(sorted_relevant_metrics) - 1:
                print(f"{round(correlations[k][model_name], 4):.4f}, ", end="")
            else:
                print(f"{round(correlations[k][model_name], 4):.4f}", end="")
        print("},")


def main() -> None:
    run_correlation_main()


if __name__ == "__main__":
    main()
