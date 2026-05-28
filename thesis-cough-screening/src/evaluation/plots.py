"""Plotting helpers for training curves and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.label_schema import TARGET_CLASSES


def plot_training_history(history: dict[str, list[float]], output_path: str | Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.get("loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train")
    axes[1].plot(history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(confusion: np.ndarray, output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(len(TARGET_CLASSES)))
    axis.set_xticklabels(TARGET_CLASSES, rotation=30, ha="right")
    axis.set_yticks(range(len(TARGET_CLASSES)))
    axis.set_yticklabels(TARGET_CLASSES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")

    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(column, row, int(confusion[row, column]), ha="center", va="center", color="black")

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
