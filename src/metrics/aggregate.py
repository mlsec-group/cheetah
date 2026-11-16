import argparse
import json
import math
import statistics
from pathlib import Path

from metrics.config import BEST_MI as BEST


def aggregate(results_list):
    if not results_list:
        return {}
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
        ret[f"{k}_mu"] = statistics.mean(vs)
        quantiles = statistics.quantiles(vs, n=10)
        ret[f"{k}_q1"] = quantiles[0]
        ret[f"{k}_q9"] = quantiles[9]
        ret[f"{k}_ci"] = 1.644850 * statistics.stdev(vs) / math.sqrt(len(vs))

    return ret


def apply_formatting(results):
    ret = ""
    for k in sorted(set(map(lambda k: k.rsplit("_", 1)[0], results.keys()))):
        mu = round(results[f"{k}_mu"]*100, 2)
        ci = round(results[f"{k}_ci"]*100, 2)
        if len(ret) > 0:
            ret += "\n"
        ret += f"{k: <4}: {mu: >5.2f}% ± {ci: >5.2f}"
    return ret


def run_aggregate(result_dir: Path) -> None:
    all_results = {}
    for metric_no in range(len(BEST)):
        results = []
        for results_path in (result_dir / str(metric_no)).glob("**/results.json"):
            with open(results_path, "r") as f:
                result = json.load(f)
            results.append(result)
        if len(results) > 0:
            all_results[metric_no] = aggregate(results)

    # cover baseline experiment (id:23)
    if len(all_results) == 0:
        results = []
        metric_no = len(BEST)
        for results_path in (result_dir / str(metric_no)).glob("**/results.json"):
            with open(results_path, "r") as f:
                result = json.load(f)
            results.append(result)
        if len(results) > 0:
            all_results[metric_no] = aggregate(results)

    with open(result_dir / "aggregated_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    if not all_results:
        print(f"No results.json files found under {result_dir}")
        return

    for i, (metric_no, aggregated_results) in enumerate(all_results.items()):
        if i > 0:
            print("-"*100)
        print(f"Metric {metric_no}:")
        print(apply_formatting(aggregated_results))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    args = p.parse_args()
    run_aggregate(args.result_dir)


if __name__ == "__main__":
    main()

