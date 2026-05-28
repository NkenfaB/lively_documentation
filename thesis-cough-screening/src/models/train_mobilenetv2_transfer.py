"""Supervised transfer-learning trainer using a MobileNetV2 backbone.

Spectrograms produced by the existing pipeline have shape (n_mels, frames, 1)
(default 64 x 256 x 1). MobileNetV2 expects 3-channel image-like inputs at
(224, 224, 3). This module wraps the spectrograms with resize + grayscale->RGB
+ MobileNetV2 preprocessing inside the model graph, so callers can feed the
existing .npy feature arrays unchanged.

Training proceeds in two optional phases:
    1. Train only the new classifier head with the backbone frozen.
    2. Optionally unfreeze the top N layers of the backbone and fine-tune
       at a lower learning rate. BatchNormalization layers are kept frozen
       during fine-tuning for stability with the small thesis dataset.

Run from the project root:
    python -m src.models.train_mobilenetv2_transfer --epochs 10 --fine-tune-epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config.paths import (
    CHECKPOINTS_DIR,
    EXPORTED_MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_TABLES_DIR,
)
from src.evaluation.plots import plot_training_history
from src.utils.io import write_json
from src.utils.logger import get_logger
from src.utils.seed import set_seed


LOGGER = get_logger(__name__)

FEATURES_DIR = PROCESSED_DATA_DIR / "features"
SCREENING_LABEL_NAMES = ("COVID", "HEALTHY_OR_NONTARGET")
COVID_INDEX = 0
HEALTHY_INDEX = 1
MOBILENET_INPUT_SIZE = (224, 224)

# Original 3-class schema (TB=0, COVID=1, HEALTHY_OR_NONTARGET=2) is collapsed
# to the binary screening schema this thesis currently supports.
ORIGINAL_TO_BINARY = {1: COVID_INDEX, 2: HEALTHY_INDEX}


def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    x_path = FEATURES_DIR / f"X_{name}.npy"
    y_path = FEATURES_DIR / f"y_{name}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing feature arrays for split '{name}'. Expected {x_path} and {y_path}. "
            "Run the preprocessing/feature pipeline before training."
        )
    return np.load(x_path), np.load(y_path)


def _prepare_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a split and collapse labels to the binary screening schema.

    Accepts either the original 3-class schema (TB=0, COVID=1, HEALTHY=2) or
    the already-binary screening schema (COVID=0, HEALTHY=1). TB rows from the
    3-class schema are dropped because the binary screening task does not use them.
    """
    x, y = _load_split(name)
    unique = set(np.unique(y).tolist())

    if unique.issubset({COVID_INDEX, HEALTHY_INDEX}):
        return x.astype(np.float32), y.astype(np.int64)

    if unique.issubset(set(ORIGINAL_TO_BINARY.keys()) | {0}):
        keep = np.isin(y, list(ORIGINAL_TO_BINARY.keys()))
        dropped = int((~keep).sum())
        if dropped:
            LOGGER.info("Split %s: dropping %d TB rows for binary screening.", name, dropped)
        x = x[keep]
        y = np.array([ORIGINAL_TO_BINARY[int(v)] for v in y[keep]], dtype=np.int64)
        return x.astype(np.float32), y

    raise ValueError(f"Split '{name}' has unrecognised label values: {sorted(unique)}")


def _log_distribution(name: str, y: np.ndarray) -> None:
    counts = np.bincount(y, minlength=len(SCREENING_LABEL_NAMES))
    parts = [f"{SCREENING_LABEL_NAMES[i]}={int(counts[i])}" for i in range(len(SCREENING_LABEL_NAMES))]
    LOGGER.info("Split %s: %d samples (%s)", name, len(y), ", ".join(parts))


def build_mobilenetv2_transfer_model(
    input_shape: tuple[int, int, int],
    num_classes: int = 2,
    dropout: float = 0.3,
    dense_units: int = 128,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build the transfer-learning model. Returns (full_model, base_model)."""
    inputs = tf.keras.Input(shape=input_shape, name="logmel_input")

    # (H, W, 1) -> (224, 224, 1) -> (224, 224, 3) -> ImageNet preprocessing
    x = tf.keras.layers.Resizing(
        MOBILENET_INPUT_SIZE[0],
        MOBILENET_INPUT_SIZE[1],
        interpolation="bilinear",
        name="resize_to_mobilenet",
    )(inputs)
    x = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([x, x, x])
    # mel features were min-max scaled to [0, 1]; MobileNetV2 expects roughly [-1, 1]
    # Two Rescaling layers replace Lambda(preprocess_input) for Keras 3 compatibility.
    # Rescaling(255) then Rescaling(1/127.5, -1) == preprocess_input: [0,1] -> [-1,1].
    # Lambda layers wrapping Python functions cannot be serialized in Keras 3, which
    # causes load_model() and TFLite export to crash.
    x = tf.keras.layers.Rescaling(255.0, name="scale_up")(x)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_norm")(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(MOBILENET_INPUT_SIZE[0], MOBILENET_INPUT_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_transfer")
    return model, base_model


def _unfreeze_top_layers(base_model: tf.keras.Model, num_layers: int) -> int:
    """Unfreeze the top `num_layers` layers, keeping BatchNorm frozen."""
    base_model.trainable = True
    total = len(base_model.layers)
    cutoff = max(total - num_layers, 0)
    unfrozen = 0
    for index, layer in enumerate(base_model.layers):
        if index < cutoff:
            layer.trainable = False
            continue
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = True
        unfrozen += 1
    return unfrozen


def _evaluate(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    batch_size: int,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    probs = model.predict(x_test, batch_size=batch_size, verbose=0)
    if probs.shape[-1] != 2:
        raise ValueError(f"Expected 2-class softmax output, got shape {probs.shape}.")

    covid_probs = probs[:, COVID_INDEX]
    default_pred = np.argmax(probs, axis=1)
    threshold_pred = np.where(covid_probs >= threshold, COVID_INDEX, HEALTHY_INDEX)

    def _metrics_block(y_pred: np.ndarray) -> dict:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[COVID_INDEX, HEALTHY_INDEX], zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=[COVID_INDEX, HEALTHY_INDEX])
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "covid_precision": float(precision[0]),
            "covid_recall": float(recall[0]),
            "covid_f1": float(f1[0]),
            "healthy_precision": float(precision[1]),
            "healthy_recall": float(recall[1]),
            "healthy_f1": float(f1[1]),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "confusion_matrix": cm.tolist(),
        }

    return {
        "threshold_covid": threshold,
        "default_argmax": _metrics_block(default_pred),
        "threshold_tuned": _metrics_block(threshold_pred),
        "num_test_samples": int(len(y_test)),
    }


def _save_confusion_plot(cm: list[list[int]], path: Path) -> None:
    import matplotlib.pyplot as plt

    cm_arr = np.array(cm)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(cm_arr, cmap="Blues")
    axis.set_xticks(range(len(SCREENING_LABEL_NAMES)))
    axis.set_xticklabels(SCREENING_LABEL_NAMES, rotation=20, ha="right")
    axis.set_yticks(range(len(SCREENING_LABEL_NAMES)))
    axis.set_yticklabels(SCREENING_LABEL_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("MobileNetV2 transfer — confusion matrix")
    for r in range(cm_arr.shape[0]):
        for c in range(cm_arr.shape[1]):
            axis.text(c, r, int(cm_arr[r, c]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.array(sorted(set(np.unique(y).tolist())))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    LOGGER.info(
        "Class weights: %s  (COVID errors penalised %.1fx more than HEALTHY)",
        {SCREENING_LABEL_NAMES[k]: round(v, 3) for k, v in cw.items()},
        cw[COVID_INDEX] / cw[HEALTHY_INDEX],
    )
    return cw


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)

    x_train, y_train = _prepare_split("train")
    x_val, y_val = _prepare_split("val")
    x_test, y_test = _prepare_split("test")

    LOGGER.info("Feature shapes: train=%s val=%s test=%s", x_train.shape, x_val.shape, x_test.shape)
    _log_distribution("train", y_train)
    _log_distribution("val", y_val)
    _log_distribution("test", y_test)

    # Balanced class weights so COVID minority class is not drowned out by healthy samples.
    class_weights = _compute_class_weights(y_train)

    model, base_model = build_mobilenetv2_transfer_model(
        input_shape=x_train.shape[1:],
        num_classes=2,
        dropout=args.dropout,
        dense_units=args.dense_units,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=LOGGER.info)

    checkpoint_path = CHECKPOINTS_DIR / "mobilenetv2_transfer.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]

    LOGGER.info("Phase 1: training classifier head with frozen backbone (%d epochs).", args.epochs)
    history_head = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    fine_tune_history = None
    if not args.skip_fine_tune and args.fine_tune_epochs > 0:
        unfrozen = _unfreeze_top_layers(base_model, args.fine_tune_layers)
        LOGGER.info(
            "Phase 2: fine-tuning top %d backbone layers (%d trainable, BN frozen) for %d epochs.",
            args.fine_tune_layers,
            unfrozen,
            args.fine_tune_epochs,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            batch_size=args.batch_size,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2,
        )

    combined_history: dict[str, list[float]] = {}
    for key, values in history_head.history.items():
        combined_history.setdefault(key, []).extend(values)
    if fine_tune_history is not None:
        for key, values in fine_tune_history.history.items():
            combined_history.setdefault(key, []).extend(values)

    history_path = REPORTS_TABLES_DIR / "mobilenetv2_transfer_history.json"
    write_json(combined_history, history_path)
    plot_training_history(combined_history, REPORTS_FIGURES_DIR / "mobilenetv2_transfer_training_history.png")

    metrics = _evaluate(model, x_test, y_test, threshold=args.threshold, batch_size=args.batch_size)
    metrics["model_name"] = "mobilenetv2_transfer"
    metrics["algorithm"] = "Supervised transfer learning (MobileNetV2 backbone) on log-mel spectrograms"
    metrics["class_weights"] = {SCREENING_LABEL_NAMES[k]: round(v, 4) for k, v in class_weights.items()}
    metrics["fine_tuned"] = bool(fine_tune_history is not None)
    metrics["fine_tune_layers"] = int(args.fine_tune_layers if fine_tune_history is not None else 0)
    metrics["epochs_head"] = int(args.epochs)
    metrics["epochs_fine_tune"] = int(args.fine_tune_epochs if fine_tune_history is not None else 0)

    metrics_path = REPORTS_TABLES_DIR / "mobilenetv2_transfer_metrics.json"
    write_json(metrics, metrics_path)
    LOGGER.info("Saved evaluation metrics to %s", metrics_path)
    LOGGER.info("Threshold-tuned summary: %s", json.dumps(metrics["threshold_tuned"], indent=2))

    _save_confusion_plot(
        metrics["threshold_tuned"]["confusion_matrix"],
        REPORTS_FIGURES_DIR / "mobilenetv2_transfer_confusion_matrix.png",
    )

    write_json(
        {
            "model_name": "mobilenetv2_transfer",
            "checkpoint": str(checkpoint_path),
            "task": "COVID vs HEALTHY_OR_NONTARGET screening",
            "threshold_covid": args.threshold,
            "class_mapping": {str(COVID_INDEX): "COVID", str(HEALTHY_INDEX): "HEALTHY_OR_NONTARGET"},
            "input_shape": list(x_train.shape[1:]),
            "mobilenet_input_shape": list(MOBILENET_INPUT_SIZE) + [3],
        },
        CHECKPOINTS_DIR / "mobilenetv2_transfer_config.json",
    )

    LOGGER.info("Saved MobileNetV2 transfer checkpoint to %s", checkpoint_path)

    # TFLite export — use SavedModel path to avoid LLVM crash in TF 2.16
    # when converting Keras models directly.
    saved_model_path = str(EXPORTED_MODELS_DIR / "mobilenetv2_screening_savedmodel")
    tflite_path = EXPORTED_MODELS_DIR / "mobilenetv2_screening.tflite"
    try:
        EXPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.export(saved_model_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        LOGGER.info("TFLite export OK: %s (%.2f MB)", tflite_path, len(tflite_model) / 1024 / 1024)
    except Exception as exc:
        LOGGER.warning("TFLite export failed: %s — checkpoint is still usable.", exc)

    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for the head-only phase.")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Epochs for the fine-tuning phase.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--fine-tune-layers", type=int, default=30, help="Top backbone layers to unfreeze.")
    parser.add_argument("--threshold", type=float, default=0.35, help="COVID decision threshold for screening mode.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-fine-tune", action="store_true", help="Train only the classifier head.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
