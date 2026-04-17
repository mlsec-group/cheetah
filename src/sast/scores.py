import argparse
import json
import re
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from collections import defaultdict


def calculate_rats_score(fpath):
    if not fpath.exists():
        return -1.0

    data = fpath.read_bytes()

    low = len(re.findall(b": Low:", data))
    medium = len(re.findall(b": Medium:", data))
    high = len(re.findall(b": High:", data))
    cum_severity = low * 1.0 + medium * 2.0 + high * 3.0

    return cum_severity


def calculate_cppcheck_score(fpath):
    if not fpath.exists():
        return -1.0

    data = fpath.read_bytes()

    error = len(re.findall(b'severity="error"', data))
    warning = len(re.findall(b'severity="warning"', data))
    style = len(re.findall(b'severity="style"', data))
    portability = len(re.findall(b'severity="portability"', data))
    information = len(re.findall(b'severity="information"', data))
    cum_severity = (
        information * 1.0
        + portability * 2.0
        + style * 3.0
        + warning * 4.0
        + error * 5.0
    )

    return cum_severity


def calculate_semgrep_scores(fpath):
    print(fpath)
    content = fpath.read_text()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON parsing failed at position {e.pos}, attempting to parse valid portion...")
        valid_content = content[:e.pos].rstrip()
        last_brace = valid_content.rfind('}')
        if last_brace != -1:
            valid_content = valid_content[:last_brace + 1]
            data = json.loads(valid_content)
        else:
            raise e

    scores = defaultdict(lambda: -1.0)
    severities = set()

    for result in data["results"]:
        idx = int(Path(result["path"]).stem)

        if scores[idx] < -0.5:
            scores[idx] = 0.0

        severity = result["extra"]["severity"]
        severities.add(severity)

        if severity == "INFO":
            scores[idx] += 1.0
        if severity == "WARNING":
            scores[idx] += 2.0
        if severity == "ERROR":
            scores[idx] += 3.0

    return scores

def run_calculate_scores(data: Path) -> None:
    hf_in_path = data / "hf"
    rats_path = data / "sast" / "rats"
    cppcheck_path = data / "sast" / "cppcheck"
    semgrep_path = data / "sast" / "semgrep"
    hf_sast_path = data / "sast" / "hf"
    hf_sast_path.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(hf_in_path)
    splits = ["train", "val", "test"]

    sast_data = defaultdict(list)

    for split in splits:
        print(f"[+] {split}")

        semgrep_fpath = semgrep_path / split / "semgrep.json"
        semgrep_scores = calculate_semgrep_scores(semgrep_fpath)

        for idx, element in tqdm(enumerate(dataset[split]), total=len(dataset[split])):
            rats_fpath = rats_path / split / str(idx)
            cppcheck_fpath = cppcheck_path / split / str(idx)

            rats_score = calculate_rats_score(rats_fpath)
            cppcheck_score = calculate_cppcheck_score(cppcheck_fpath)

            element["rats"] = rats_score
            element["cppcheck"] = cppcheck_score
            element["semgrep"] = semgrep_scores[idx]
            sast_data[split].append(element)

        sast_data[split] = Dataset.from_list(sast_data[split])

    sast_dataset = DatasetDict(sast_data)
    sast_dataset.save_to_disk(hf_sast_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", type=Path, required=True)
    args = p.parse_args()
    run_calculate_scores(args.data)


if __name__ == "__main__":
    main()
