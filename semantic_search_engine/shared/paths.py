from pathlib import Path


def default_config_path() -> Path:
    package_dir = Path(__file__).resolve().parents[1]
    repository_root = package_dir.parent
    development_config = package_dir / "config" / "default.yaml"
    if (repository_root / "dataset_builder").is_dir():
        return development_config
    return package_dir / "config" / "installed.yaml"
