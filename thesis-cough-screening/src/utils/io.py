"""Basic file IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    df.to_csv(target, index=False)


def discover_tabular_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    patterns = ("*.csv", "*.tsv", "*.txt", "*.json")
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root_path.rglob(pattern))
    return sorted(path for path in matches if path.is_file())
