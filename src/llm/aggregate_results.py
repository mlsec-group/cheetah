import argparse
import math
import json
import statistics

from pathlib import Path


def collect_results(path, selection_predicate):
    if not path.is_dir():
        return []

    ret = []
    for p in path.iterdir():
        assert p.name.isnumeric()
        if (p / "results.json").exists():
            with open(p / "results.json", "r", encoding="utf8") as f:
                ret.append(json.load(f))
            continue
        if (p / "trained_model.pkl").exists():
            continue
        selected_path = None
        for c in p.iterdir():
            if c.name == "runs":
                continue
            assert "checkpoint" in c.name
            if not (c / "config.json").exists():
                continue
            if not ((c / "model.safetensors").exists() or (c / "model.safetensors.index.json").exists()):
                continue
            if selected_path is None:
                selected_path = c
                continue
            selected_c_num = int(selected_path.name.rsplit("-")[-1])
            c_num = int(c.name.rsplit("-")[-1])
            if selection_predicate(c_num, selected_c_num):
                selected_path = c
        if selected_path is not None and (c / "results.json").exists():
            with open(selected_path / "results.json", "r", encoding="utf8") as f:
                ret.append(json.load(f))

    return ret


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
        ret[f"{k}_mu"] = statistics.mean(vs)
        quantiles = statistics.quantiles(vs, n=10)
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


def run_llm_aggregate(models_path: Path, last_checkpoint: bool = False, dat: str | None = None) -> None:
    if not models_path.is_dir():
        raise ValueError("Provided path should be a directory containing all checkpoints to be tested in respective folders")

    selection_predicate = lambda x1, x2: x1 < x2
    if last_checkpoint:
        selection_predicate = lambda x1, x2: x1 > x2

    results = collect_results(models_path, selection_predicate)
    print(f"Found results for {len(results)} models")

    aggregated_results = aggregate(results)
    if dat is not None:
        for result in results:
            print(100 * result[dat])
    print(apply_formatting(aggregated_results))


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate LLM runs.")
    p.add_argument("models_path", type=Path)
    p.add_argument("--last-checkpoint", action="store_true")
    p.add_argument("--dat", default=None)
    args = p.parse_args()
    run_llm_aggregate(args.models_path, last_checkpoint=args.last_checkpoint, dat=args.dat)


if __name__ == "__main__":
    main()
