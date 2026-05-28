"""Resampling and amplitude normalization helpers."""

from __future__ import annotations

import librosa
import numpy as np


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if original_sr == target_sr:
        return audio.astype(np.float32), original_sr
    resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return resampled.astype(np.float32), target_sr


def peak_normalize(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < eps:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)
