"""Train the baseline cough classification model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config.label_schema import LABEL_TO_INDEX, TARGET_CLASSES
from src.config.paths import CHECKPOINTS_DIR, METADATA_SPLITS_DIR, PROCESSED_DATA_DIR, REPORTS_FIGURES_DIR
from src.config.settings import load_settings
from src.features.feature_utils import build_feature_arrays
from src.models.baseline_cnn import build_baseline_cnn
from src.evaluation.plots import plot_training_history
from src.utils.io import write_json
from src.utils.logger import get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)


def _load_split_csv(split_name: str) -> pd.DataFrame:
    path = METADATA_SPLITS_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def _load_precomputed(split_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    root = PROCESSED_DATA_DIR / "features"
    x_path = root / f"X_{split_name}.npy"
    y_path = root / f"y_{split_name}.npy"
    if x_path.exists() and y_path.exists():
        LOGGER.info("Using precomputed features for split '%s'.", split_name)
        return np.load(x_path), np.load(y_path)
    return None


def _build_or_load_features(split_name: str, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    cached = _load_precomputed(split_name)
    if cached is not None:
        x, y = cached
        if max_samples is not None:
            return x[:max_samples], y[:max_samples]
        return x, y

    settings = load_settings()
    split_df = _load_split_csv(split_name)
    x, y, _ = build_feature_arrays(split_df, settings=settings, limit=max_samples)
    return x, y


def _require_all_classes(labels: np.ndarray) -> None:
    present = {TARGET_CLASSES[index] for index in np.unique(labels).tolist()}
    missing = [label for label in TARGET_CLASSES if label not in present]
    if missing:
        raise ValueError(
            "Training requires all 3 target classes. Missing classes: "
            f"{', '.join(missing)}. Complete the manual TB acquisition and rerun metadata preparation."
        )


def train_baseline() -> Path:
    settings = load_settings()
    set_seed(settings.splits.random_state)

    x_train, y_train = _build_or_load_features("train", settings.training.max_train_samples)
    x_val, y_val = _build_or_load_features("val", settings.training.max_eval_samples)
    # Remap labels from 3-class [0=TB, 1=COVID, 2=HEALTHY] to 2-class [0=COVID, 1=HEALTHY]
    print(f"Original labels - unique values: {np.unique(y_train)}, counts: {np.bincount(y_train)}")
    label_map = {1: 0, 2: 1}  # Map COVID(1)->0, HEALTHY(2)->1, ignore TB(0)
    y_train = np.array([label_map[label] for label in y_train])
    y_val = np.array([label_map[label] for label in y_val])
    print(f"Remapped labels - unique values: {np.unique(y_train)}, counts: {np.bincount(y_train)}")


    #     _require_all_classes(y_train)

    model = build_baseline_cnn(input_shape=x_train.shape[1:], num_classes=len(np.unique(y_train)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings.training.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = CHECKPOINTS_DIR / "baseline_cnn.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=settings.training.validation_monitor,
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=settings.training.validation_monitor,
            patience=settings.training.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=settings.training.batch_size,
        epochs=settings.training.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training_history(history.history, REPORTS_FIGURES_DIR / "training_history.png")
    write_json(history.history, CHECKPOINTS_DIR / "training_history.json")
    write_json({"label_to_index": LABEL_TO_INDEX}, CHECKPOINTS_DIR / "label_map.json")
    LOGGER.info("Saved baseline checkpoint to %s", checkpoint_path)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    train_baseline()


if __name__ == "__main__":
    main()
