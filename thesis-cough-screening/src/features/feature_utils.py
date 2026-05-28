"""Shared feature extraction utilities and optional preprocessing CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.label_schema import LABEL_TO_INDEX, TARGET_CLASSES
from src.config.paths import METADATA_SPLITS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.config.settings import ProjectSettings, load_settings
from src.features.mel_spectrogram import compute_log_mel_spectrogram
from src.preprocessing.audio_loading import load_audio
from src.preprocessing.noise_reduction import reduce_noise
from src.preprocessing.quality_checks import quality_flag
from src.preprocessing.resample_normalize import peak_normalize
from src.preprocessing.segmentation import extract_loudest_segment, trim_silence
from src.utils.io import ensure_dir
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def pad_or_truncate(feature: np.ndarray, max_frames: int) -> np.ndarray:
    current_frames = feature.shape[1]
    if current_frames == max_frames:
        return feature.astype(np.float32)
    if current_frames > max_frames:
        return feature[:, :max_frames].astype(np.float32)

    pad_width = max_frames - current_frames
    return np.pad(feature, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)


def min_max_scale(feature: np.ndarray) -> np.ndarray:
    min_value = float(np.min(feature))
    max_value = float(np.max(feature))
    if max_value - min_value < 1e-8:
        return np.zeros_like(feature, dtype=np.float32)
    scaled = (feature - min_value) / (max_value - min_value)
    return scaled.astype(np.float32)


def preprocess_audio_file(path: str | Path, settings: ProjectSettings) -> tuple[np.ndarray, int, str]:
    audio, sample_rate = load_audio(path, target_sr=settings.audio.target_sample_rate, mono=settings.audio.mono)
    if settings.audio.trim_silence:
        audio = trim_silence(audio)
    audio = extract_loudest_segment(audio, max_duration_samples=int(settings.audio.max_duration_sec * sample_rate))
    audio = reduce_noise(audio, enabled=False)
    if settings.audio.peak_normalize:
        audio = peak_normalize(audio)
    flag = quality_flag(
        audio,
        sample_rate,
        min_duration_sec=settings.audio.min_duration_sec,
        max_duration_sec=settings.audio.max_duration_sec,
    )
    return audio, sample_rate, flag


def build_mel_input(path: str | Path, settings: ProjectSettings) -> tuple[np.ndarray, str]:
    audio, sample_rate, flag = preprocess_audio_file(path, settings)
    feature = compute_log_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_mels=settings.features.n_mels,
        n_fft=settings.features.n_fft,
        hop_length=settings.features.hop_length,
        win_length=settings.features.win_length,
        fmin=settings.features.fmin,
        fmax=settings.features.fmax,
    )
    feature = pad_or_truncate(feature, settings.features.max_frames)
    feature = min_max_scale(feature)
    feature = np.expand_dims(feature, axis=-1)
    return feature.astype(np.float32), flag


def build_feature_arrays(
    split_df: pd.DataFrame,
    settings: ProjectSettings,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = split_df.copy()
    if limit is not None:
        rows = rows.head(limit).copy()

    features: list[np.ndarray] = []
    labels: list[int] = []
    kept_rows: list[dict] = []

    for row in tqdm(rows.to_dict(orient="records"), desc="Extracting features"):
        path = PROJECT_ROOT / str(row["file_path"])
        if not path.exists():
            continue
        try:
            feature, flag = build_mel_input(path, settings)
        except Exception as error:
            LOGGER.warning("Skipping %s because feature extraction failed: %s", path, error)
            continue

        row["quality_flag"] = flag
        if flag != "OK":
            LOGGER.info("Keeping %s with quality flag '%s' for traceability.", row["sample_id"], flag)
        label = row.get("label_multiclass")
        if label not in LABEL_TO_INDEX:
            continue
        features.append(feature)
        labels.append(LABEL_TO_INDEX[label])
        kept_rows.append(row)

    if not features:
        raise ValueError("No features were generated from the provided metadata.")

    feature_array = np.stack(features).astype(np.float32)
    label_array = np.array(labels, dtype=np.int64)
    kept_df = pd.DataFrame(kept_rows)
    return feature_array, label_array, kept_df


def save_split_features(
    split_name: str,
    split_csv: Path,
    output_dir: Path,
    settings: ProjectSettings,
    limit: int | None = None,
) -> None:
    split_df = pd.read_csv(split_csv)
    features, labels, kept_df = build_feature_arrays(split_df, settings=settings, limit=limit)
    ensure_dir(output_dir)
    np.save(output_dir / f"X_{split_name}.npy", features)
    np.save(output_dir / f"y_{split_name}.npy", labels)
    kept_df.to_csv(output_dir / f"metadata_{split_name}.csv", index=False)
    LOGGER.info("Saved precomputed features for split '%s' to %s", split_name, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default=str(METADATA_SPLITS_DIR))
    parser.add_argument("--output-dir", default=str(PROCESSED_DATA_DIR / "features"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    for split_name in ("train", "val", "test"):
        split_csv = splits_dir / f"{split_name}.csv"
        if not split_csv.exists():
            LOGGER.warning("Split file missing, skipping preprocessing: %s", split_csv)
            continue
        save_split_features(split_name, split_csv, output_dir, settings, limit=args.limit)

    LOGGER.info("Supported target classes: %s", ", ".join(TARGET_CLASSES))


if __name__ == "__main__":
    main()
