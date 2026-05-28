"""Project settings with optional YAML overrides."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioSettings:
    target_sample_rate: int = 16_000
    mono: bool = True
    peak_normalize: bool = True
    trim_silence: bool = True
    min_duration_sec: float = 0.4
    max_duration_sec: float = 12.0


@dataclass
class FeatureSettings:
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 512
    fmin: int = 50
    fmax: int = 8_000
    max_frames: int = 256
    n_mfcc: int = 20


@dataclass
class SplitSettings:
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


@dataclass
class TrainingSettings:
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 1e-3
    validation_monitor: str = "val_loss"
    early_stopping_patience: int = 5
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


@dataclass
class ExportSettings:
    enable_optimizations: bool = False


@dataclass
class ProjectSettings:
    audio: AudioSettings = field(default_factory=AudioSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    splits: SplitSettings = field(default_factory=SplitSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    export: ExportSettings = field(default_factory=ExportSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SETTINGS = ProjectSettings()


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(path: str | Path | None = None) -> ProjectSettings:
    """Load settings from YAML if provided, otherwise return defaults."""
    candidate = path or os.getenv("THESIS_SETTINGS_PATH")
    if not candidate:
        return DEFAULT_SETTINGS

    config_path = Path(candidate).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}

    merged = _deep_update(DEFAULT_SETTINGS.to_dict(), override)
    return ProjectSettings(
        audio=AudioSettings(**merged["audio"]),
        features=FeatureSettings(**merged["features"]),
        splits=SplitSettings(**merged["splits"]),
        training=TrainingSettings(**merged["training"]),
        export=ExportSettings(**merged["export"]),
    )


def save_default_settings(path: str | Path) -> None:
    config_path = Path(path)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(DEFAULT_SETTINGS.to_dict(), handle, sort_keys=False)
