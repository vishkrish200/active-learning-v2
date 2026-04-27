from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray
    n_windows: int
    n_files: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype="float32")
        return ((array - self.mean) / self.scale).astype("float32")

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "n_windows": int(self.n_windows),
            "n_files": int(self.n_files),
        }


def fit_feature_scaler(
    paths: Sequence[Path],
    *,
    feature_dim: int,
    max_files: int | None = None,
    max_windows_per_file: int | None = None,
) -> FeatureScaler:
    selected_paths = _bounded_paths(paths, max_files=max_files)
    if not selected_paths:
        raise ValueError("Cannot fit feature scaler without feature files.")

    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    n_windows = 0
    n_files = 0
    for path in selected_paths:
        with np.load(path) as data:
            values = np.asarray(data["window_features"], dtype="float32")
        if values.ndim != 2 or values.shape[1] != feature_dim:
            continue
        values = _bounded_windows(values, max_windows_per_file=max_windows_per_file)
        if len(values) == 0:
            continue
        finite_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        total += np.sum(finite_values, axis=0, dtype=np.float64)
        total_sq += np.sum(finite_values * finite_values, axis=0, dtype=np.float64)
        n_windows += int(len(finite_values))
        n_files += 1

    if n_windows == 0:
        raise ValueError("Feature scaler found no valid window_features arrays.")
    mean = total / n_windows
    variance = np.maximum(total_sq / n_windows - mean * mean, 1.0e-6)
    scale = np.sqrt(variance)
    scale = np.where(scale < 1.0e-3, 1.0, scale)
    return FeatureScaler(
        mean=mean.astype("float32"),
        scale=scale.astype("float32"),
        n_windows=n_windows,
        n_files=n_files,
    )


def load_feature_scaler(payload: dict[str, object] | None, *, feature_dim: int) -> FeatureScaler | None:
    if not payload:
        return None
    mean = np.asarray(payload.get("mean", []), dtype="float32")
    scale = np.asarray(payload.get("scale", []), dtype="float32")
    if mean.shape != (feature_dim,) or scale.shape != (feature_dim,):
        raise ValueError("Checkpoint feature scaler shape does not match feature_dim.")
    scale = np.where(scale < 1.0e-6, 1.0, scale).astype("float32")
    return FeatureScaler(
        mean=mean,
        scale=scale,
        n_windows=int(payload.get("n_windows", 0)),
        n_files=int(payload.get("n_files", 0)),
    )


def _bounded_paths(paths: Sequence[Path], *, max_files: int | None) -> list[Path]:
    path_list = list(paths)
    if max_files is None or max_files <= 0 or len(path_list) <= max_files:
        return path_list
    indices = np.linspace(0, len(path_list) - 1, num=int(max_files), dtype=int)
    return [path_list[int(index)] for index in indices]


def _bounded_windows(values: np.ndarray, *, max_windows_per_file: int | None) -> np.ndarray:
    if max_windows_per_file is None or max_windows_per_file <= 0 or len(values) <= max_windows_per_file:
        return values
    indices = np.linspace(0, len(values) - 1, num=int(max_windows_per_file), dtype=int)
    return values[indices]
