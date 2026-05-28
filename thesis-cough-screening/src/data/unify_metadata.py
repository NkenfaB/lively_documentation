"""Clean per-dataset metadata and build a unified metadata table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf

from src.config.label_schema import label_binary_from_multiclass, map_label
from src.config.paths import DATASET_DIRS, METADATA_CLEAN_DIR, PROJECT_ROOT, ensure_project_dirs
from src.data.audit_metadata import AUDIO_EXTENSIONS, choose_metadata_file, find_audio_files, load_tabular_file
from src.data.dataset_registry import get_dataset_names
from src.utils.io import discover_tabular_files, write_table
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
STANDARD_COLUMNS = [
    "sample_id",
    "subject_id",
    "dataset_name",
    "file_path",
    "label_raw",
    "label_binary",
    "label_multiclass",
    "age",
    "gender",
    "country",
    "recording_type",
    "sample_rate",
    "duration_sec",
    "quality_flag",
    "split",
    "notes",
    "include_for_training",
]


def _relative_to_project(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _safe_audio_info(path: Path) -> tuple[int | None, float | None]:
    try:
        info = sf.info(path)
        return info.samplerate, round(float(info.duration), 3)
    except RuntimeError:
        return None, None


def _pick_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lowered = {column.lower(): column for column in df.columns.astype(str)}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _build_audio_index(dataset_name: str, dataset_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in find_audio_files(dataset_dir):
        sample_rate, duration = _safe_audio_info(path)
        stem = path.stem
        subject_id = path.parent.name if dataset_name == "coswara" else None
        recording_type = "cough" if "cough" in stem.lower() else "unknown"
        rows.append(
            {
                "sample_id": stem,
                "subject_id": subject_id,
                "dataset_name": dataset_name,
                "file_path": _relative_to_project(path),
                "recording_type": recording_type,
                "sample_rate": sample_rate,
                "duration_sec": duration,
            }
        )
    return pd.DataFrame(rows)


def _load_metadata_table(dataset_dir: Path) -> pd.DataFrame:
    candidates = discover_tabular_files(dataset_dir)
    selected = choose_metadata_file(candidates)
    if selected is None:
        return pd.DataFrame()
    return load_tabular_file(selected)


def _normalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def _apply_common_fields(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for column in STANDARD_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df["dataset_name"] = dataset_name
    df["split"] = df["split"].fillna("")
    df["quality_flag"] = df["quality_flag"].fillna("UNCHECKED")
    df["notes"] = df["notes"].fillna("")
    return df[STANDARD_COLUMNS]


def _attach_labels(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    multiclass: list[str | None] = []
    notes: list[str] = []
    for row in df.to_dict(orient="records"):
        label, note = map_label(dataset_name, row)
        multiclass.append(label)
        notes.append(note if not row.get("notes") else f"{row['notes']} | {note}")

    df["label_multiclass"] = multiclass
    df["label_binary"] = [label_binary_from_multiclass(label) for label in multiclass]
    df["notes"] = notes
    df["include_for_training"] = df["label_multiclass"].notna()
    return df


def clean_coswara_metadata(dataset_dir: Path) -> pd.DataFrame:
    audio_df = _build_audio_index("coswara", dataset_dir)
    if audio_df.empty:
        return _apply_common_fields(audio_df, "coswara")

    audio_df = audio_df[audio_df["file_path"].str.contains("cough", case=False, na=False)].copy()
    metadata_df = _normalize_metadata_columns(_load_metadata_table(dataset_dir))

    if not metadata_df.empty:
        subject_column = _pick_column(metadata_df, ["id", "user_id", "subject_id", "participant_id"])
        age_column = _pick_column(metadata_df, ["age", "age_group"])
        gender_column = _pick_column(metadata_df, ["gender", "sex"])
        country_column = _pick_column(metadata_df, ["country", "location"])
        covid_column = _pick_column(metadata_df, ["covid_status", "health_status", "status"])

        if subject_column:
            metadata_df = metadata_df.rename(columns={subject_column: "subject_id"})
        if age_column:
            metadata_df = metadata_df.rename(columns={age_column: "age"})
        if gender_column:
            metadata_df = metadata_df.rename(columns={gender_column: "gender"})
        if country_column:
            metadata_df = metadata_df.rename(columns={country_column: "country"})
        if covid_column:
            metadata_df = metadata_df.rename(columns={covid_column: "label_raw"})

        keep = [column for column in ["subject_id", "age", "gender", "country", "label_raw"] if column in metadata_df]
        if "subject_id" in keep:
            audio_df = audio_df.merge(metadata_df[keep].drop_duplicates(), on="subject_id", how="left")

    return _apply_common_fields(_attach_labels(audio_df, "coswara"), "coswara")


def clean_coughvid_metadata(dataset_dir: Path) -> pd.DataFrame:
    audio_df = _build_audio_index("coughvid", dataset_dir)
    if audio_df.empty:
        return _apply_common_fields(audio_df, "coughvid")

    metadata_df = _normalize_metadata_columns(_load_metadata_table(dataset_dir))
    if not metadata_df.empty:
        sample_column = _pick_column(metadata_df, ["uuid", "file_name", "filename", "sample_id", "cough_id"])
        subject_column = _pick_column(metadata_df, ["subject_id", "participant_id", "user_id"])
        status_column = _pick_column(metadata_df, ["status", "covid_status", "assessment_result"])
        age_column = _pick_column(metadata_df, ["age", "age_group"])
        gender_column = _pick_column(metadata_df, ["gender", "sex"])
        country_column = _pick_column(metadata_df, ["country", "location"])

        rename_map = {}
        if sample_column:
            rename_map[sample_column] = "sample_id"
        if subject_column:
            rename_map[subject_column] = "subject_id"
        if status_column:
            rename_map[status_column] = "label_raw"
        if age_column:
            rename_map[age_column] = "age"
        if gender_column:
            rename_map[gender_column] = "gender"
        if country_column:
            rename_map[country_column] = "country"

        metadata_df = metadata_df.rename(columns=rename_map)
        if "sample_id" in metadata_df:
            metadata_df["sample_id"] = metadata_df["sample_id"].astype(str).str.replace(r"\.[a-z0-9]+$", "", regex=True)

        keep = [column for column in ["sample_id", "subject_id", "age", "gender", "country", "label_raw"] if column in metadata_df]
        if "sample_id" in keep:
            audio_df = audio_df.merge(metadata_df[keep].drop_duplicates(subset=["sample_id"]), on="sample_id", how="left")

    audio_df["recording_type"] = "cough"
    return _apply_common_fields(_attach_labels(audio_df, "coughvid"), "coughvid")


def clean_tb_metadata(dataset_dir: Path) -> pd.DataFrame:
    audio_df = _build_audio_index("tb", dataset_dir)
    metadata_df = _normalize_metadata_columns(_load_metadata_table(dataset_dir))

    if audio_df.empty and metadata_df.empty:
        empty = pd.DataFrame(columns=STANDARD_COLUMNS)
        return empty

    if not metadata_df.empty:
        sample_column = _pick_column(metadata_df, ["sample_id", "recording_id", "file_name", "filename"])
        subject_column = _pick_column(metadata_df, ["subject_id", "participant_id", "patient_id"])
        status_column = _pick_column(metadata_df, ["tb_status", "diagnosis", "reference_diagnosis"])
        age_column = _pick_column(metadata_df, ["age", "age_group"])
        gender_column = _pick_column(metadata_df, ["gender", "sex"])
        country_column = _pick_column(metadata_df, ["country", "site", "location"])

        rename_map = {}
        if sample_column:
            rename_map[sample_column] = "sample_id"
        if subject_column:
            rename_map[subject_column] = "subject_id"
        if status_column:
            rename_map[status_column] = "label_raw"
        if age_column:
            rename_map[age_column] = "age"
        if gender_column:
            rename_map[gender_column] = "gender"
        if country_column:
            rename_map[country_column] = "country"
        metadata_df = metadata_df.rename(columns=rename_map)

        keep = [column for column in ["sample_id", "subject_id", "age", "gender", "country", "label_raw"] if column in metadata_df]
        if not audio_df.empty and "sample_id" in metadata_df:
            audio_df = audio_df.merge(metadata_df[keep].drop_duplicates(subset=["sample_id"]), on="sample_id", how="left")
        elif keep:
            audio_df = metadata_df[keep].copy()

    if "recording_type" not in audio_df:
        audio_df["recording_type"] = "cough"
    if "file_path" not in audio_df:
        audio_df["file_path"] = None
    if "sample_rate" not in audio_df:
        audio_df["sample_rate"] = None
    if "duration_sec" not in audio_df:
        audio_df["duration_sec"] = None

    return _apply_common_fields(_attach_labels(audio_df, "tb"), "tb")


def clean_dataset_metadata(dataset_name: str) -> pd.DataFrame:
    dataset_dir = DATASET_DIRS[dataset_name]
    if dataset_name == "coswara":
        return clean_coswara_metadata(dataset_dir)
    if dataset_name == "coughvid":
        return clean_coughvid_metadata(dataset_dir)
    if dataset_name == "tb":
        return clean_tb_metadata(dataset_dir)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_unified_metadata(dataset_names: list[str] | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    requested = dataset_names or get_dataset_names()
    per_dataset_frames: list[pd.DataFrame] = []

    for dataset_name in requested:
        df = clean_dataset_metadata(dataset_name)
        target = METADATA_CLEAN_DIR / f"{dataset_name}_metadata.csv"
        write_table(df, target)
        LOGGER.info("Saved cleaned metadata: %s", target)
        per_dataset_frames.append(df)

    unified = pd.concat(per_dataset_frames, ignore_index=True) if per_dataset_frames else pd.DataFrame(columns=STANDARD_COLUMNS)
    unified = unified.sort_values(["dataset_name", "sample_id"], na_position="last")
    write_table(unified, METADATA_CLEAN_DIR / "unified_metadata.csv")
    LOGGER.info("Saved unified metadata: %s", METADATA_CLEAN_DIR / "unified_metadata.csv")
    return unified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=get_dataset_names(), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [args.dataset] if args.dataset else None
    build_unified_metadata(names)


if __name__ == "__main__":
    main()
