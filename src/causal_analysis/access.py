
import json
import math
import statistics

from pathlib import Path


relevant_metrics = ["C1", "C2", "C3", "C4", "V1", "V3", "V6", "V7", "V8", "V10", "x1", "x2", "x3", "x4", "t1", "t2", "t3", "s1", "s2", "s3", "s4", "m1", "m2"]

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


def load_all_r2s(full_path: Path):
    paths = full_path.rglob("r2_results.json")
    results = []
    for path in paths:
        run = int(str(path).replace(str(full_path), "")[1:].split("/", 1)[0])
        if run > 10:
            continue
        if "checkpoint" in str(path.parent):
            # check if smallest checkpoint
            model_root = path.parent.parent
            smallest_path = min((p for p in model_root.iterdir()), key=lambda p: int(str(p).split("checkpoint-")[-1]))
            if path.parent != smallest_path:
                continue
        with open(path, "r", encoding="utf8") as f:
            results.append(json.load(f))
    return results


def aggregate(results_list):
    if len(results_list) == 1:
        ret = {}
        results = results_list[0]
        for k, v in results.items():
            ret[f"{k}_mu"] = v
            ret[f"{k}_ci"] = float("inf")
        return ret

    all_results = {k: [] for k in results_list[0].keys()}
    for results in results_list:
        for k,v in results.items():
            all_results[k].append(v)

    ret = {}
    for k, vs in all_results.items():
        vs = [max(v, 0) for v in vs]
        ret[f"{k}_mu"] = statistics.mean(vs)
        ret[f"{k}_ci"] = 1.644850 * statistics.stdev(vs) / math.sqrt(len(vs))

    return ret


def run_access_r2_aggregate() -> None:
    models = Path(".") / "hf-model"
    unixcoder_r2s = load_all_r2s(models / "microsoft__unixcoder-base-nine")
    pdbert_r2s = load_all_r2s(models / "data__pdbert")
    codebert_r2s = load_all_r2s(models / "microsoft__codebert-base")
    starcoder_r2s = load_all_r2s(models / "bigcode__starcoder2-7b")
    codegen_r2s = load_all_r2s(models / "Salesforce__codegen25-7b-multi_P")

    all_results = {
        "unixcoder": aggregate(unixcoder_r2s[:]),
        "pdbert": aggregate(pdbert_r2s[:]),
        "codebert": aggregate(codebert_r2s[:]),
        "codegen": aggregate(codegen_r2s[:]),
        "starcoder": aggregate(starcoder_r2s[:]),
    }

    print("r^2 (information access)")
    print(",".join([MAPPING[k] for k in sorted_relevant_metrics]))
    for model_name, results in all_results.items():
        print(model_name)
        print("{", end="")
        for ik, k in enumerate(sorted_relevant_metrics):
            if ik < len(relevant_metrics) - 1:
                print(f"{round(results[k + '_mu'], 4):.4f}, ", end="")
            else:
                print(f"{round(results[k + '_mu'], 4):.4f}", end="")
        print("},")


def main() -> None:
    run_access_r2_aggregate()


if __name__ == "__main__":
    main()
