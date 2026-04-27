from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from marginal_value.preprocessing.quality import load_modal_jsonl_imu


DEFAULT_WINDOW_SAMPLES = 300
DEFAULT_STRIDE_SAMPLES = 150
FEATURE_DIM = 75


def window_feature_matrix_from_jsonl(
    path: str | Path,
    *,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    stride_samples: int = DEFAULT_STRIDE_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    samples, _timestamps = load_modal_jsonl_imu(path)
    window_features = compute_window_feature_matrix(
        samples,
        window_samples=window_samples,
        stride_samples=stride_samples,
    )
    clip_features = np.mean(window_features, axis=0)
    return window_features, clip_features


def compute_window_feature_matrix(
    samples: np.ndarray,
    *,
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    stride_samples: int = DEFAULT_STRIDE_SAMPLES,
) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError("samples must be [n_samples, >=6] with accel and gyro channels.")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive.")
    if stride_samples <= 0:
        raise ValueError("stride_samples must be positive.")

    windows = _window_slices(len(values), window_samples=window_samples, stride_samples=stride_samples)
    rows = [_window_feature_row(values[start:end, :6]) for start, end in windows]
    return np.asarray(rows, dtype=np.float32)


def _window_slices(n_samples: int, *, window_samples: int, stride_samples: int) -> list[tuple[int, int]]:
    if n_samples <= 0:
        raise ValueError("samples must not be empty.")
    if n_samples <= window_samples:
        return [(0, n_samples)]
    starts = list(range(0, n_samples - window_samples + 1, stride_samples))
    if starts[-1] != n_samples - window_samples:
        starts.append(n_samples - window_samples)
    return [(start, start + window_samples) for start in starts]


def _window_feature_row(window: np.ndarray) -> np.ndarray:
    acc = window[:, :3]
    gyro = window[:, 3:6]
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    acc_jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1) * 30.0
    gyro_jerk = np.linalg.norm(np.diff(gyro, axis=0), axis=1) * 30.0

    signals = (acc_norm, gyro_norm, acc_jerk, gyro_jerk)
    features: list[float] = []
    for signal in signals:
        features.extend(_distribution_stats(signal))
        features.extend(_spectral_stats(signal))
    features.extend(_window_context_features(acc_norm, gyro_norm, acc, gyro))

    if len(features) != FEATURE_DIM:
        raise RuntimeError(f"Expected {FEATURE_DIM} features, produced {len(features)}.")
    return np.asarray(features, dtype=np.float32)


def _distribution_stats(signal: Sequence[float]) -> list[float]:
    values = np.asarray(signal, dtype=float)
    if values.size == 0:
        values = np.asarray([0.0], dtype=float)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.sqrt(np.mean(values * values))),
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(values, 5)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 75)),
        float(np.percentile(values, 95)),
    ]


def _spectral_stats(signal: Sequence[float]) -> list[float]:
    values = np.asarray(signal, dtype=float)
    if values.size < 3:
        return [0.0] * 8
    centered = values - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(values.size, d=1.0 / 30.0)
    if spectrum.size > 1:
        spectrum = spectrum[1:]
        freqs = freqs[1:]
    total = float(np.sum(spectrum))
    if total <= 1.0e-12:
        return [0.0] * 8
    probs = spectrum / total
    centroid = float(np.sum(freqs * probs))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * probs)))
    dominant_index = int(np.argmax(spectrum))
    dominant_freq = float(freqs[dominant_index])
    dominant_fraction = float(spectrum[dominant_index] / total)
    entropy = float(-np.sum(probs * np.log(probs + 1.0e-12)) / np.log(len(probs) + 1.0e-12))
    low = float(np.sum(spectrum[freqs < 1.0]) / total)
    mid = float(np.sum(spectrum[(freqs >= 1.0) & (freqs < 5.0)]) / total)
    high = float(np.sum(spectrum[freqs >= 5.0]) / total)
    return [total / float(values.size), dominant_freq, dominant_fraction, entropy, centroid, spread, low, mid + high]


def _window_context_features(acc_norm: np.ndarray, gyro_norm: np.ndarray, acc: np.ndarray, gyro: np.ndarray) -> list[float]:
    stationary = (np.abs(acc_norm - np.median(acc_norm)) < 0.05) & (gyro_norm < 0.02)
    acc_imbalance = _axis_imbalance(acc)
    gyro_imbalance = _axis_imbalance(gyro)
    return [float(np.mean(stationary)), acc_imbalance, gyro_imbalance]


def _axis_imbalance(values: np.ndarray) -> float:
    std = np.std(values, axis=0)
    return float(np.max(std) / (np.median(std) + 1.0e-6))
