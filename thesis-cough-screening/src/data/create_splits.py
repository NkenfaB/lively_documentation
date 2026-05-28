"""Create train, validation, and test splits from unified metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.paths import METADATA_CLEAN_DIR, METADATA_SPLITS_DIR, PROJECT_ROOT
from src.config.settings import load_settings
from src.utils.io import write_table
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _filter_trainable_rows(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["include_for_training"].fillna(False)].copy()
    if "file_path" in filtered.columns:
        filtered = filtered[filtered["file_path"].notna()].copy()
        filtered["file_exists"] = filtered["file_path"].apply(lambda value: (PROJECT_ROOT / str(value)).exists())
        filtered = filtered[filtered["file_exists"]].drop(columns=["file_exists"])
    return filtered


def _sample_level_split(df: pd.DataFrame, random_state: int, train_size: float, val_size: float, test_size: float):
    stratify = df["label_multiclass"] if df["label_multiclass"].nunique() > 1 else None
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    adjusted_val = val_size / (train_size + val_size)
    stratify_train_val = train_val["label_multiclass"] if train_val["label_multiclass"].nunique() > 1 else None
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        random_state=random_state,
        stratify=stratify_train_val,
    )
    return train, val, test, "sample_level"


def _group_level_split(df: pd.DataFrame, random_state: int, train_size: float, val_size: float, test_size: float):
    working = df.copy()
    working["group_id"] = working["dataset_name"].astype(str) + "::" + working["subject_id"].astype(str)
    groups = working.groupby("group_id", dropna=False).agg(
        label_multiclass=("label_multiclass", "first"),
        dataset_name=("dataset_name", "first"),
    ).reset_index()

    class_counts = groups["label_multiclass"].value_counts()
    if class_counts.empty or class_counts.min() < 2:
        return _sample_level_split(df, random_state, train_size, val_size, test_size)

    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=random_state,
        stratify=groups["label_multiclass"],
    )
    adjusted_val = val_size / (train_size + val_size)
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=adjusted_val,
        random_state=random_state,
        stratify=train_val_groups["label_multiclass"],
    )

    train = working[working["group_id"].isin(train_groups["group_id"])].drop(columns=["group_id"])
    val = working[working["group_id"].isin(val_groups["group_id"])].drop(columns=["group_id"])
    test = working[working["group_id"].isin(test_groups["group_id"])].drop(columns=["group_id"])
    return train, val, test, "subject_independent"


def create_splits(metadata_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    settings = load_settings()
    source = Path(metadata_path) if metadata_path else METADATA_CLEAN_DIR / "unified_metadata.csv"
    df = pd.read_csv(source)
    trainable = _filter_trainable_rows(df)

    if trainable.empty:
        raise ValueError(
            "No trainable rows found. Run downloads/audit/unify steps first and ensure labels map cleanly."
        )

    has_subjects = trainable["subject_id"].notna().any()
    if has_subjects:
        train, val, test, strategy = _group_level_split(
            trainable,
            random_state=settings.splits.random_state,
            train_size=settings.splits.train_size,
            val_size=settings.splits.val_size,
            test_size=settings.splits.test_size,
        )
    else:
        train, val, test, strategy = _sample_level_split(
            trainable,
            random_state=settings.splits.random_state,
            train_size=settings.splits.train_size,
            val_size=settings.splits.val_size,
            test_size=settings.splits.test_size,
        )

    train = train.assign(split="train")
    val = val.assign(split="val")
    test = test.assign(split="test")

    split_frames = {"train": train, "val": val, "test": test}
    for split_name, split_df in split_frames.items():
        target = METADATA_SPLITS_DIR / f"{split_name}.csv"
        write_table(split_df, target)
        LOGGER.info("Saved %s split to %s", split_name, target)

    combined = pd.concat([train, val, test], ignore_index=True)
    write_table(combined, METADATA_SPLITS_DIR / "unified_with_splits.csv")
    LOGGER.info("Split strategy used: %s", strategy)
    return split_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_splits(args.metadata_path)


if __name__ == "__main__":
    main()
