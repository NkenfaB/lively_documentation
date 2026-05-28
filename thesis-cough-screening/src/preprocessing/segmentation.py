"""Simple silence trimming and cough-like event segmentation."""

from __future__ import annotations

import librosa
import numpy as np


def trim_silence(audio: np.ndarray, top_db: int = 25) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def detect_active_segments(
    audio: np.ndarray,
    frame_length: int = 1024,
    hop_length: int = 256,
    energy_quantile: float = 0.75,
) -> list[tuple[int, int]]:
    if audio.size == 0:
        return []

    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = float(np.quantile(energy, energy_quantile))
    active = energy >= threshold
    segments: list[tuple[int, int]] = []
    start = None

    for frame_index, is_active in enumerate(active):
        if is_active and start is None:
            start = frame_index
        if not is_active and start is not None:
            end = frame_index
            segments.append((start * hop_length, end * hop_length + frame_length))
            start = None

    if start is not None:
        segments.append((start * hop_length, len(audio)))
    return segments


def extract_loudest_segment(audio: np.ndarray, max_duration_samples: int | None = None) -> np.ndarray:
    segments = detect_active_segments(audio)
    if not segments:
        return audio.astype(np.float32)

    best_start, best_end = max(segments, key=lambda segment: segment[1] - segment[0])
    excerpt = audio[best_start:best_end]
    if max_duration_samples is not None and excerpt.size > max_duration_samples:
        excerpt = excerpt[:max_duration_samples]
    return excerpt.astype(np.float32)
