"""MFCC feature extraction."""

from __future__ import annotations

import librosa
import numpy as np


def compute_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfcc.astype(np.float32)
