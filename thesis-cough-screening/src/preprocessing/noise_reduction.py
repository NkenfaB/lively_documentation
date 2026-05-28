"""Basic noise reduction placeholder."""

from __future__ import annotations

import numpy as np
from scipy.signal import wiener


def reduce_noise(audio: np.ndarray, enabled: bool = False) -> np.ndarray:
    if not enabled or audio.size == 0:
        return audio.astype(np.float32)
    return wiener(audio).astype(np.float32)
