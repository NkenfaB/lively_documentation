"""Central path definitions used across the project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

METADATA_DIR = PROJECT_ROOT / "metadata"
METADATA_RAW_DIR = METADATA_DIR / "raw"
METADATA_CLEAN_DIR = METADATA_DIR / "cleaned"
METADATA_SPLITS_DIR = METADATA_DIR / "splits"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"
REPORTS_LOGS_DIR = REPORTS_DIR / "logs"

DATASET_DIRS = {
    "coswara": RAW_DATA_DIR / "coswara",
    "coughvid": RAW_DATA_DIR / "coughvid",
    "tb": RAW_DATA_DIR / "tb",
}


def ensure_project_dirs() -> None:
    """Create the core project directories if they do not exist."""
    for path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        METADATA_RAW_DIR,
        METADATA_CLEAN_DIR,
        METADATA_SPLITS_DIR,
        CHECKPOINTS_DIR,
        EXPORTED_MODELS_DIR,
        REPORTS_FIGURES_DIR,
        REPORTS_TABLES_DIR,
        REPORTS_LOGS_DIR,
        *DATASET_DIRS.values(),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root when needed."""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
