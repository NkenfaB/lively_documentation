import numpy as np
import soundfile as sf

from src.preprocessing.audio_loading import load_audio


def test_load_audio_roundtrip(tmp_path):
    sample_rate = 16_000
    time = np.linspace(0, 0.25, int(sample_rate * 0.25), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * time)
    wav_path = tmp_path / "tone.wav"
    sf.write(wav_path, audio, sample_rate)

    loaded, loaded_sr = load_audio(wav_path, target_sr=sample_rate, mono=True)

    assert loaded_sr == sample_rate
    assert loaded.ndim == 1
    assert loaded.size == audio.size
