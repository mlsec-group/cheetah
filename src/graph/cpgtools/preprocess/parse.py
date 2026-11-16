import argparse
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from loguru import logger as log

from graph.cpgtools.paths import joern_scripts_dir
from graph.cpgtools.util import ParallelProgress, code_hash


def parse_cpg(code: str, path: str, out_dir: Path, joern_memory: str, joern_cores: int):
    h = code_hash(code)
    out_file = out_dir / path
    out_file = out_file.with_suffix(".cpg")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists():
        return

    with NamedTemporaryFile(suffix=".c") as f:
        f.write(code.encode())
        f.flush()
        joern_parse_script = joern_scripts_dir() / "joern_parse.sh"
        result = subprocess.run(
            ["bash", str(joern_parse_script), str(f.name), str(out_file), str(joern_memory), str(joern_cores)],
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        log.error(result.stderr)
        return h, result.stderr


def parse_cpgs(source_dir: Path, out_dir: Path, report: Path,
               n_jobs: int = 1,
               joern_memory: str = "10g", joern_cores: int = 1):
    source_files = read_source_files(source_dir)
    args = [(code, path, out_dir, joern_memory, joern_cores) for path, code in source_files.items()]
    errors = ParallelProgress(n_jobs=n_jobs, task="Parsing CPG").run(parse_cpg, args)
    errors = {h: e for h, e in filter(lambda e: e is not None and e != "", errors)}
    report.parent.mkdir(exist_ok=True, parents=True)
    with open(report, "w") as f:
        json.dump(errors, f, indent=4)


def read_source_files(source_dir: Path) -> dict[str, str]:
    # TODO: change this if you e.g. read from jsonl files
    source_files = list(source_dir.glob("**/*.c"))
    return {str(file.relative_to(source_dir)): file.read_text() for file in source_files}


def main() -> None:
    p = argparse.ArgumentParser(description="Parse sources into CPGs.")
    p.add_argument("source_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--report", type=Path, default=Path("data/reports/parse.json"))
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--joern-memory", default="10g")
    p.add_argument("--joern-cores", type=int, default=1)
    args = p.parse_args()
    parse_cpgs(
        args.source_dir,
        args.out_dir,
        args.report,
        n_jobs=args.n_jobs,
        joern_memory=args.joern_memory,
        joern_cores=args.joern_cores,
    )


if __name__ == "__main__":
    main()
