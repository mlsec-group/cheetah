from pathlib import Path


def repo_root() -> Path:
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "scripts" / "config.sh").is_file():
        return candidate
    for parent in Path(__file__).resolve().parents:
        if (parent / "scripts" / "config.sh").is_file():
            return parent
    raise RuntimeError("Could not locate repository root (expected scripts/config.sh)")


def data_dir() -> Path:
    return repo_root() / "data"


def pdbert_dir() -> Path:
    return data_dir() / "pdbert"


def is_pdbert_path(model: str) -> bool:
    path = Path(model)
    if path.name != "pdbert":
        return False
    try:
        return path.resolve() == pdbert_dir().resolve()
    except OSError:
        return False


def model_slug(model: str) -> str:
    if is_pdbert_path(model):
        return "data__pdbert"
    return model.replace("/", "__")


def resolve_model_path(model: str) -> str:
    if is_pdbert_path(model):
        return str(pdbert_dir())
    path = Path(model)
    if path.is_dir():
        return str(path.resolve())
    return model
