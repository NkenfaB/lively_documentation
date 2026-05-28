"""Mel spectrogram feature extraction."""

from __future__ import annotations

import librosa
import numpy as np


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 512,
    fmin: int = 50,
    fmax: int = 8_000,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)
