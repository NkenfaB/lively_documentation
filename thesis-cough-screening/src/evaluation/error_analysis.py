"""Evaluate a trained model and save summary artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config.label_schema import TARGET_CLASSES
from src.config.paths import CHECKPOINTS_DIR, METADATA_SPLITS_DIR, REPORTS_FIGURES_DIR, REPORTS_TABLES_DIR
from src.config.settings import load_settings
from src.evaluation.metrics import build_confusion_matrix, compute_metrics, save_metrics_json, save_metrics_table
from src.evaluation.plots import plot_confusion_matrix
from src.features.feature_utils import build_feature_arrays
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def evaluate_model(checkpoint: str | Path | None = None) -> dict[str, float | dict]:
    settings = load_settings()
    model_path = Path(checkpoint) if checkpoint else CHECKPOINTS_DIR / "baseline_cnn.keras"
    split_path = METADATA_SPLITS_DIR / "test.csv"

    if not split_path.exists():
        raise FileNotFoundError(f"Missing test split: {split_path}")

    split_df = pd.read_csv(split_path)
    x_test, y_test, kept_df = build_feature_arrays(
        split_df,
        settings=settings,
        limit=settings.training.max_eval_samples,
    )

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    metrics = compute_metrics(y_test, y_pred)
    confusion = build_confusion_matrix(y_test, y_pred)

    save_metrics_table(metrics, REPORTS_TABLES_DIR / "test_metrics.csv")
    save_metrics_json(metrics, REPORTS_TABLES_DIR / "test_metrics.json")
    plot_confusion_matrix(confusion, REPORTS_FIGURES_DIR / "confusion_matrix.png")

    analysis_df = kept_df.copy()
    analysis_df["true_label"] = [TARGET_CLASSES[index] for index in y_test]
    analysis_df["predicted_label"] = [TARGET_CLASSES[index] for index in y_pred]
    analysis_df["correct"] = analysis_df["true_label"] == analysis_df["predicted_label"]
    analysis_df.to_csv(REPORTS_TABLES_DIR / "test_predictions.csv", index=False)
    analysis_df[~analysis_df["correct"]].to_csv(REPORTS_TABLES_DIR / "misclassified_examples.csv", index=False)

    LOGGER.info("Saved evaluation artifacts to reports/figures and reports/tables.")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(args.checkpoint)


if __name__ == "__main__":
    main()
