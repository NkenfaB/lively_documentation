"""Evaluation metric helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.config.label_schema import TARGET_CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | dict]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(TARGET_CLASSES))),
            target_names=list(TARGET_CLASSES),
            zero_division=0,
            output_dict=True,
        ),
    }


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_CLASSES))))


def save_metrics_table(metrics: dict[str, float | dict], path: str | Path) -> None:
    rows = [
        {"metric": "accuracy", "value": metrics["accuracy"]},
        {"metric": "precision_macro", "value": metrics["precision_macro"]},
        {"metric": "recall_macro", "value": metrics["recall_macro"]},
        {"metric": "f1_macro", "value": metrics["f1_macro"]},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def save_metrics_json(metrics: dict[str, float | dict], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
