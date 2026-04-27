from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from marginal_value.data.split_manifest import SplitSample


def load_window_features(path: str | Path) -> np.ndarray:
    with np.load(Path(path)) as data:
        if "window_features" not in data:
            raise KeyError("Feature file must contain a 'window_features' array.")
        return np.asarray(data["window_features"], dtype=np.float32)


def extract_patch_vectors(
    window_features: np.ndarray,
    *,
    patch_size_windows: int,
    stride_windows: int,
) -> np.ndarray:
    values = np.asarray(window_features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("window_features must be a 2D [n_windows, n_features] array.")
    if patch_size_windows <= 0:
        raise ValueError("patch_size_windows must be positive.")
    if stride_windows <= 0:
        raise ValueError("stride_windows must be positive.")
    if len(values) == 0:
        raise ValueError("window_features must contain at least one window.")

    padded = _pad_to_patch_size(values, patch_size_windows)
    starts = list(range(0, len(padded) - patch_size_windows + 1, stride_windows))
    if starts[-1] != len(padded) - patch_size_windows:
        starts.append(len(padded) - patch_size_windows)
    patches = [padded[start : start + patch_size_windows].reshape(-1) for start in starts]
    return np.asarray(patches, dtype=np.float32)


def patch_vectors_for_rows(
    rows: Iterable[SplitSample],
    *,
    patch_size_windows: int,
    stride_windows: int,
    max_patches_per_row: int | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    patch_batches: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for row in rows:
        features = load_window_features(row.feature_path)
        patches = extract_patch_vectors(
            features,
            patch_size_windows=patch_size_windows,
            stride_windows=stride_windows,
        )
        if max_patches_per_row is not None:
            if max_patches_per_row <= 0:
                raise ValueError("max_patches_per_row must be positive when provided.")
            patches = patches[:max_patches_per_row]
        patch_batches.append(patches)
        for patch_index in range(len(patches)):
            metadata.append(
                {
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "patch_index": patch_index,
                    "feature_path": str(row.feature_path),
                }
            )

    if not patch_batches:
        raise ValueError("No rows were provided for patch extraction.")
    return np.vstack(patch_batches), metadata


def _pad_to_patch_size(values: np.ndarray, patch_size_windows: int) -> np.ndarray:
    if len(values) >= patch_size_windows:
        return values
    pad_count = patch_size_windows - len(values)
    padding = np.repeat(values[-1:, :], pad_count, axis=0)
    return np.concatenate([values, padding], axis=0)

