"""Run inference on a single cough audio file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config.label_schema import TARGET_CLASSES
from src.config.paths import CHECKPOINTS_DIR
from src.config.settings import load_settings
from src.features.feature_utils import build_mel_input


def predict_file(audio_path: str | Path, checkpoint: str | Path | None = None) -> dict[str, float]:
    model_path = Path(checkpoint) if checkpoint else CHECKPOINTS_DIR / "baseline_cnn.keras"
    model = tf.keras.models.load_model(model_path)
    feature, _ = build_mel_input(audio_path, load_settings())
    batch = np.expand_dims(feature, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    return {label: float(probabilities[index]) for index, label in enumerate(TARGET_CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_file(args.audio_path, args.checkpoint)
    for label, probability in predictions.items():
        print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()
