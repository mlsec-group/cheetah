import json
import numpy as np

from pathlib import Path


def load_all(path: Path):
    paths = path.rglob("concatenation_results.json")
    results = []
    for path in paths:
        if "checkpoint" in str(path.parent):
            # check if smallest checkpoint
            model_root = path.parent.parent
            smallest_path = min((p for p in model_root.iterdir()), key=lambda p: int(str(p).split("checkpoint-")[-1]))
            if path.parent != smallest_path:
                continue
        with open(path, "r", encoding="utf8") as f:
            results.append(json.load(f))
    return results

def run_cross_f1_diff() -> None:
    models = Path(".") / "hf-model"
    unixcoder = load_all(models / "microsoft__unixcoder-base-nine")
    pdbert = load_all(models / "data__pdbert")
    codebert = load_all(models / "microsoft__codebert-base")
    starcoder = load_all(models / "bigcode__starcoder2-7b")
    codegen = load_all(models / "Salesforce__codegen25-7b-multi_P")

    # in percentage points
    print("Difference in F1-Score between concatenation and only embedding (Percentage points!)")
    print("UniXcoder, PDBert, LineVul, StarCoder 2, CodeGen 2.5")
    print(
        round(sum(t[1] - t[0] for t in unixcoder) / len(unixcoder) * 100, 2),
        round(sum(t[1] - t[0] for t in pdbert) / len(pdbert) * 100, 2),
        round(sum(t[1] - t[0] for t in codebert) / len(codebert) * 100, 2),
        round(sum(t[1] - t[0] for t in starcoder) / len(starcoder) * 100, 2),
        round(sum(t[1] - t[0] for t in codegen) / len(codegen) * 100, 2),
    )


def main() -> None:
    run_cross_f1_diff()


if __name__ == "__main__":
    main()
