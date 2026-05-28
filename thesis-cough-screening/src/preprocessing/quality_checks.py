"""Simple audio quality checks for thesis-friendly debugging."""

from __future__ import annotations

import numpy as np


def audio_duration_sec(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(len(audio) / sample_rate)


def clipping_ratio(audio: np.ndarray, threshold: float = 0.999) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


def silence_ratio(audio: np.ndarray, threshold: float = 1e-3) -> float:
    if audio.size == 0:
        return 1.0
    return float(np.mean(np.abs(audio) < threshold))


def quality_flag(
    audio: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.4,
    max_duration_sec: float = 12.0,
) -> str:
    duration = audio_duration_sec(audio, sample_rate)
    if duration < min_duration_sec:
        return "TOO_SHORT"
    if duration > max_duration_sec:
        return "TOO_LONG"
    if clipping_ratio(audio) > 0.05:
        return "CLIPPED"
    if silence_ratio(audio) > 0.98:
        return "MOSTLY_SILENT"
    return "OK"
