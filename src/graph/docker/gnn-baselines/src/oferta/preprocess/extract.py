import json
from pathlib import Path
import hashlib


def extract(in_train: Path, in_val: Path, in_test: Path, out_dir: Path):
    for in_file, which in ((in_train, "train"), (in_val, "val"), (in_test, "test")):
        with open(in_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)

                    project = row["project"]
                    func = row["func"]
                    target = row["target"]
                    commit = row.get("commit_id")
                    if commit is None:
                        commit = "unknown"
                    h = hashlib.sha256(func.encode("utf-8")).hexdigest()

                    out_file = out_dir / which / project / commit / f"{target}_{h}.c"
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_file, "w") as f:
                        f.write(func)
