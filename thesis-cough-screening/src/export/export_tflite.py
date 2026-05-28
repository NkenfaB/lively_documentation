"""Export a trained TensorFlow model to TensorFlow Lite."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from src.config.paths import CHECKPOINTS_DIR, EXPORTED_MODELS_DIR
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def export_tflite(checkpoint: str | Path, output: str | Path) -> Path:
    model = tf.keras.models.load_model(checkpoint)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # TODO: Add representative datasets and post-training quantization.
    # TODO: Benchmark on target mobile hardware before final deployment.
    tflite_model = converter.convert()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    LOGGER.info("Saved TFLite model to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "baseline_cnn.keras"))
    parser.add_argument("--output", default=str(EXPORTED_MODELS_DIR / "baseline_cnn.tflite"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_tflite(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
