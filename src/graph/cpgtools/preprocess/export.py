import argparse
import os
from pathlib import Path
from shutil import rmtree
from subprocess import PIPE, run
from tempfile import TemporaryDirectory
from typing import List

from loguru import logger as log

from graph.cpgtools.paths import joern_scripts_dir
from graph.cpgtools.util import ParallelProgress


def export_cpg(in_files: List[Path], out_dir: Path, in_root: Path, joern_memory: str, joern_cores: int):
    in_args = []
    out_args = []
    for in_file in in_files:
        out_file = out_dir / in_file.relative_to(in_root).parent / f"{in_file.stem}.json"
        if out_file.exists():
            continue
        out_file.parent.mkdir(exist_ok=True, parents=True)
        in_args.append(str(in_file.absolute()))
        out_args.append(str(out_file.absolute()))
    in_args = ",".join(in_args)
    out_args = ",".join(out_args)

    if len(in_args) == 0:
        return []

    errors = []
    export_script = joern_scripts_dir() / "export.sc"
    run_script = joern_scripts_dir() / "run.sh"
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        workspace = Path(tmp_dir) / "workspace" / in_files[0].name
        workspace.mkdir(exist_ok=True, parents=True)
        cmd = [
            str(run_script),
            str(joern_memory),
            str(joern_cores),
            str(export_script),
            f'"{in_args}"',
            f'"{out_args}"'
        ]
        env = os.environ.copy()
        env["TERM"] = "dumb"
        result = run(cmd, stdout=PIPE, stderr=PIPE, cwd=workspace, text=True, env=env)
        rmtree(workspace, ignore_errors=True)
        if len(result.stderr) > 0:
            log.error(result.stderr)
            errors.append(result.stderr)
        for in_file, out_file in zip(in_files, out_args.split(",")):
            if not Path(out_file).exists():
                errors.append(str(in_file))
    return errors


def export_cpgs(
        cpg_dir: Path,
        out_dir: Path,
        report: Path,
        n_jobs: int,
        joern_memory: str,
        joern_cores: int,
        batch_size: int = 100
):
    args = batched_args(cpg_dir.absolute(), out_dir.absolute(), batch_size, joern_memory, joern_cores)
    errors = ParallelProgress(n_jobs=n_jobs, task="Exporting CPGs").run(export_cpg, args)
    report.parent.mkdir(exist_ok=True, parents=True)
    with open(report, "w") as f:
        for batch in errors:
            for line in batch:
                f.write(line)
                f.write("\n")


def batched_args(
        cpg_dir: Path,
        out_dir: Path,
        batch_size: int,
        joern_memory: str,
        joern_cores: int
):
    args = []
    cpg_files = list(cpg_dir.glob("**/*.cpg"))
    for i in range(0, len(cpg_files), batch_size):
        in_files = cpg_files[i:i + batch_size]
        args.append((in_files, out_dir, cpg_dir, joern_memory, joern_cores))
    return args


def export_num_batches(cpg_dir: Path, batch_size: int = 100):
    cpg_files = list(cpg_dir.glob("**/*.cpg"))
    return len(cpg_files) // batch_size + (1 if len(cpg_files) % batch_size > 0 else 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Export CPGs to JSON.")
    p.add_argument("cpg_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--report", type=Path, default=Path("data/reports/export.json"))
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--joern-memory", default="10g")
    p.add_argument("--joern-cores", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=100)
    args = p.parse_args()
    export_cpgs(
        args.cpg_dir,
        args.out_dir,
        args.report,
        args.n_jobs,
        args.joern_memory,
        args.joern_cores,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
