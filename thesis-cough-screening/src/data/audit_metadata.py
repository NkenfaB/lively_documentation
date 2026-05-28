"""Audit raw datasets and summarize metadata availability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf

from src.config.paths import DATASET_DIRS, METADATA_RAW_DIR, ensure_project_dirs
from src.data.dataset_registry import get_dataset_names
from src.utils.io import discover_tabular_files, write_json
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def infer_recording_type(path: Path) -> str:
    stem = path.stem.lower()
    if "cough" in stem:
        return "cough"
    if "breath" in stem:
        return "breath"
    if "voice" in stem or "speech" in stem:
        return "voice"
    return "unknown"


def find_audio_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported tabular file: {path}")


def choose_metadata_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    preferred_terms = ("metadata", "label", "annotation", "participant", "clinical", "compiled")
    ranked = sorted(
        paths,
        key=lambda path: (
            sum(term in path.name.lower() for term in preferred_terms),
            -len(path.name),
        ),
        reverse=True,
    )
    return ranked[0]


def probe_audio_metadata(paths: list[Path], max_files: int = 25) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths[:max_files]:
        try:
            info = sf.info(path)
            rows.append(
                {
                    "file": str(path),
                    "sample_rate": info.samplerate,
                    "duration_sec": round(info.duration, 3),
                    "channels": info.channels,
                    "recording_type": infer_recording_type(path),
                }
            )
        except RuntimeError as error:
            rows.append({"file": str(path), "error": str(error)})
    return rows


def audit_dataset(dataset_name: str) -> dict[str, Any]:
    ensure_project_dirs()
    dataset_dir = DATASET_DIRS[dataset_name]
    audio_files = find_audio_files(dataset_dir)
    metadata_candidates = discover_tabular_files(dataset_dir)
    selected = choose_metadata_file(metadata_candidates)

    summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "audio_file_count": len(audio_files),
        "metadata_candidate_count": len(metadata_candidates),
        "metadata_candidates": [str(path) for path in metadata_candidates[:20]],
        "selected_metadata_file": str(selected) if selected else None,
        "audio_probe": probe_audio_metadata(audio_files),
        "selected_metadata_columns": [],
    }

    if selected:
        try:
            df = load_tabular_file(selected)
            summary["selected_metadata_columns"] = df.columns.astype(str).tolist()
            summary["selected_metadata_rows"] = int(len(df))
        except Exception as error:
            summary["selected_metadata_error"] = str(error)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=get_dataset_names(), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [args.dataset] if args.dataset else get_dataset_names()
    for dataset_name in dataset_names:
        summary = audit_dataset(dataset_name)
        target = METADATA_RAW_DIR / f"{dataset_name}_audit.json"
        write_json(summary, target)
        LOGGER.info("Saved audit summary: %s", target)


if __name__ == "__main__":
    main()
