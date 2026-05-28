"""Audio loading helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(Path(path), sr=target_sr, mono=mono)
    return audio.astype(np.float32), int(sample_rate)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=0).astype(np.float32)
