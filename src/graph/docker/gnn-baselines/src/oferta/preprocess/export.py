import os
from typing import List
from importlib.resources import files
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

from gnn-baselines.util import ParallelProgress
from subprocess import Popen, PIPE


def export_cpg(in_files: List[Path], out_dir: Path, in_root: Path, joern_bin: Path):
    in_args = []
    out_args = []
    for in_file in in_files:
        out_file = out_dir / in_file.relative_to(in_root).parent / f"{in_file.stem}.json"
        if out_file.exists():
            continue
        out_file.parent.mkdir(exist_ok=True, parents=True)
        in_args.append(str(in_file))
        out_args.append(str(out_file))
    in_args = ",".join(in_args)
    out_args = ",".join(out_args)

    if len(in_args) == 0:
        return []

    errors = []
    with TemporaryDirectory() as tmp_dir, files("gnn-baselines.preprocess") / "cpg_batch_export.sc" as export_script:
        os.chdir(tmp_dir)
        workspace = Path(tmp_dir) / "workspace" / in_files[0].name
        workspace.mkdir(exist_ok=True, parents=True)
        p = Popen([
            joern_bin,
            "--script", export_script,
            "--param", f"cpgFileArg={in_args}",
            "--param", f"outFileArg={out_args}"
        ], stdout=PIPE, stderr=PIPE, cwd=workspace)
        stdout, stderr = p.communicate()
        rmtree(workspace, ignore_errors=True)
        if len(stderr) > 0:
            errors.append(stderr.decode())
        for in_file, out_file in zip(in_files, out_args.split(",")):
            if not Path(out_file).exists():
                errors.append(str(in_file))
    return errors


def export_cpgs(in_dir: Path, out_dir: Path, report: Path, n_jobs: int, joern_bin: Path, batch_size: int = 100):
    in_files = list(in_dir.glob("**/*.cpg"))
    in_batches = [in_files[i:i + batch_size] for i in range(0, len(in_files), batch_size)]
    args = [(in_batch, out_dir, in_dir, joern_bin) for in_batch in in_batches]
    errors = ParallelProgress(n_jobs=n_jobs, task="Exporting CPGs").run(export_cpg, args)
    report.parent.mkdir(exist_ok=True, parents=True)
    with open(report, "w") as f:
        for batch in errors:
            for line in batch:
                f.write(line)
                f.write("\n")
