import json
from pathlib import Path
from importlib.resources import files
import subprocess

from gnn-baselines.util import ParallelProgress


def parse_cpg(in_file: Path, out_dir: Path, in_root: Path):
    out_file = out_dir / in_file.relative_to(in_root).parent / f"{in_file.stem}.cpg"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists():
        return

    with files('gnn-baselines.preprocess') / 'joern_parse.sh' as joern_parse_script:
        result = subprocess.run(
            ['bash', str(joern_parse_script), str(in_file), str(out_file)],
            capture_output=True,
            text=True
        )

    if result.returncode != 0:
        return (in_file, result.stderr)


def parse_cpgs(in_dir: Path, out_dir: Path, report: Path, n_jobs: int = 1):
    in_files = list(in_dir.glob("**/*.c"))
    args = [(in_file, out_dir, in_dir) for in_file in in_files]
    errors = ParallelProgress(n_jobs=n_jobs, task="Parsing CPG").run(parse_cpg, args)
    errors = {str(in_file): error for in_file, error in filter(lambda e: e is not None, errors)}
    report.parent.mkdir(exist_ok=True, parents=True)
    with open(report, "w") as f:
        json.dump(errors, f, indent=4)
