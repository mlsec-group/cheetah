import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_FILE = "test_results.json"


def collect_results(result_root: Path):

    all_results = []
    for result_path in result_root.glob(f"**/{TARGET_FILE}"):
        train_config_path = result_path.parent / "train_config.json"

        with open(result_path, "r") as f:
            results = json.load(f)
        with open(train_config_path, "r") as f:
            train_config = json.load(f)

        all_results.append({
            "model_name": train_config["model_name"],
            "task_id": train_config["task_id"],
            **results
        })
    return all_results


def _p90(observations):
    """Compute the 90% confidence interval. """
    if len(observations) < 2:
        return 0.0
    return 1.645*(observations.std()/np.sqrt(len(observations)))


def process_results(df_raw, out_file, verbose=False):
    df = df_raw.rename(columns={
        "model_name": "Model",
        # "F1": "F1",
        "Prec": "Precision",
        "Rec": "Recall",
        "auprec": "AUPREC",
        "vd_score": "VD_S"
    })
    
    out_rows = []
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        if verbose:
            print(f"\n{model}")
        
        for metric in ["F1", "Precision", "Recall"]:
            mean = model_df[metric].mean()
            p90_val = _p90(model_df[metric])
            if verbose:
                print(f"{metric}: {mean:.3f} ± {p90_val:.3f}")
            out_rows.append({"model_name": model, "metric": metric, "mean": mean, "p90_val": p90_val})
    with open(out_file, "w") as f:
        json.dump(out_rows, f, indent=4)


def collect(result_root: Path, out_file: Path, verbose: bool = False):
    results = collect_results(result_root)
    df = pd.DataFrame(results)
    process_results(df, out_file, verbose)


def main() -> None:
    p = argparse.ArgumentParser(description="Collect and summarize GNN eval results.")
    p.add_argument("result_root", type=Path)
    p.add_argument("out_file", type=Path)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    collect(args.result_root, args.out_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
