from pathlib import Path
from collections import defaultdict
import json
import csv
from subprocess import Popen, PIPE
from typing import Dict
from time import sleep, perf_counter

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn

from gnn-baselines.datasets.diversevul import DiverseVul


def clean_url(url: str):
    if url.startswith("https://cgit.freedesktop.org/xorg/lib/"):
        project = url[len("https://cgit.freedesktop.org/xorg/lib/"):].split("/")[0]
        url = f"https://gitlab.freedesktop.org/xorg/lib/{project}.git"
    if "/commit" in url:
        url = url[:url.find("/commit")]

    return url


class PrimeVul(DiverseVul):
    _name = "primevul"
    _num_classes = 2

    @staticmethod
    def preprocess_meta(raw_dir: Path, urls_csv: Path, commits_csv: Path):
        urls = {}
        commits = defaultdict(set)
        for in_file in raw_dir.glob("**/*.jsonl"):
            with open(in_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    project = data["project"]
                    url = data["project_url"]
                    url = clean_url(url)
                    urls[project] = url
                    commits[project].add(data["commit_id"])

        urls_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(urls_csv, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["project", "url"])
            for project, url in urls.items():
                writer.writerow([project, url])

        commits_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(commits_csv, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["project", "commit"])
            for project, commits in commits.items():
                for commit in commits:
                    writer.writerow([project, commit])

    @staticmethod
    def clone_repos(urls_csv: Path, override_urls_csv: Path, repo_root: Path, report: Path):
        urls = {}
        with open(urls_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                urls[row["project"]] = row["url"]
        with open(override_urls_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                urls[row["project"]] = row["url"]
        failed = clone_projects(urls, repo_root)
        if failed:
            report.parent.mkdir(parents=True, exist_ok=True)
            with open(report, "w") as f:
                json.dump(failed, f, indent=4)

    @staticmethod
    def find_first_versions(urls_csv: Path, commits_csv: Path):
        pass


def clone(url: str, target_dir: Path, timeout: int):
    p = Popen(["timeout", str(timeout), "git", "clone", url, target_dir], stdout=PIPE, stderr=PIPE)
    _, stderr = p.communicate()
    if p.returncode > 0:
        logger.warning(f"{url} failed with exit code {p.returncode}, error:\n{stderr.decode()}")
        return url


def clone_projects(clone_urls: Dict[str, str], clone_root: Path, max_per_second=1, timeout=600):
    min_duration = 1 / max_per_second
    start = None
    end = None
    failed = []
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        for project, url in progress.track(clone_urls.items(), description=f"Cloning {len(clone_urls)} repos"):
            repo_dir = clone_root / project
            repo_dir.parent.mkdir(parents=True, exist_ok=True)

            if (repo_dir).exists():
                failed.append(url)
                continue

            # limit clone rate (avoid getting blocked)
            if start and end and min_duration < end - start:
                sleep(end - start - min_duration)
            if not repo_dir.exists():
                start = perf_counter()
                failed_url = clone(url, repo_dir, timeout)
                if failed_url:
                    failed.append(failed_url)
                    start = None
                end = perf_counter()
    return failed


def validate_repos(urls: Dict[str, str], clone_root: Path):
    failed = []
    for project, url in urls.items():
        if not (clone_root / project).exists() or len(list((clone_root / project).iterdir())) == 0:
            failed.append(url)
    return failed
