from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from marginal_value.logging_utils import log_event, log_progress


def _safe_array(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2:
        raise ValueError("IMU samples must be a 2D array shaped [time, channels]")
    if array.shape[1] < 6:
        raise ValueError("IMU samples must include at least six channels")
    return array


def _fraction_bad_timestamps(timestamps: np.ndarray | None, sample_rate: float) -> tuple[float, float]:
    if timestamps is None or len(timestamps) < 3:
        return 0.0, 0.0

    ts = np.asarray(timestamps, dtype=float)
    diffs = np.diff(ts)
    finite = np.isfinite(diffs)
    if not np.any(finite):
        return 1.0, 1.0

    expected = 1.0 / sample_rate
    repeated = float(np.mean(diffs[finite] <= 0.0))
    jitter = float(np.mean(np.abs(diffs[finite] - expected) > expected * 0.50))
    return repeated, jitter


def compute_quality_features(
    samples: np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
    sample_rate: float = 30.0,
    flatline_variance_epsilon: float = 1.0e-8,
    saturation_abs_threshold: float = 250.0,
    spike_abs_threshold: float = 100.0,
) -> dict[str, float]:
    """Return artifact-oriented quality features for one worker sequence."""

    array = _safe_array(samples)
    n_samples = array.shape[0]
    finite_mask = np.isfinite(array)
    nan_fraction = float(np.mean(np.isnan(array)))
    inf_fraction = float(np.mean(np.isinf(array)))
    missing_rate = float(1.0 - np.mean(finite_mask))

    filled = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    variances = np.var(filled, axis=0)
    flatline_fraction = float(np.mean(variances <= flatline_variance_epsilon))
    saturation_fraction = float(np.mean(np.abs(filled) >= saturation_abs_threshold))
    max_abs_value = float(np.max(np.abs(filled[:, :6]))) if n_samples else 0.0
    extreme_value_fraction = float(np.mean(np.abs(filled[:, :6]) >= spike_abs_threshold))

    diffs = np.diff(filled, axis=0)
    if diffs.size:
        robust_scale = np.median(np.abs(diffs - np.median(diffs, axis=0)), axis=0) + 1.0e-6
        robust_spikes = np.abs(diffs) > (25.0 * robust_scale)
        absolute_spikes = np.abs(diffs) > spike_abs_threshold
        spike_rate = float(np.mean(robust_spikes | absolute_spikes))
        high_frequency_energy = float(np.mean(diffs * diffs))
    else:
        spike_rate = 0.0
        high_frequency_energy = 0.0

    acc = filled[:, :3]
    gyro = filled[:, 3:6]
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    stationary = (np.abs(acc_norm - np.nanmedian(acc_norm)) < 0.05) & (gyro_norm < 0.02)
    stationary_fraction = float(np.mean(stationary)) if n_samples else 1.0

    axis_std = np.std(filled[:, :6], axis=0)
    axis_imbalance = float(np.max(axis_std) / (np.median(axis_std) + 1.0e-6))
    repeated_timestamp_fraction, timestamp_jitter_fraction = _fraction_bad_timestamps(
        timestamps, sample_rate
    )

    features = {
        "n_samples": float(n_samples),
        "duration_sec": float(n_samples / sample_rate),
        "missing_rate": missing_rate,
        "nan_fraction": nan_fraction,
        "inf_fraction": inf_fraction,
        "flatline_fraction": flatline_fraction,
        "saturation_fraction": saturation_fraction,
        "max_abs_value": max_abs_value,
        "extreme_value_fraction": extreme_value_fraction,
        "spike_rate": spike_rate,
        "high_frequency_energy": high_frequency_energy,
        "stationary_fraction": stationary_fraction,
        "axis_imbalance": axis_imbalance,
        "repeated_timestamp_fraction": repeated_timestamp_fraction,
        "timestamp_jitter_fraction": timestamp_jitter_fraction,
    }
    features["quality_score"] = score_quality(features)
    return features


def compute_quality_from_jsonl(
    path: str | Path,
    *,
    sample_rate: float = 30.0,
    max_samples: int | None = None,
) -> dict[str, float]:
    samples, timestamps = load_modal_jsonl_imu(path, max_samples=max_samples)
    return compute_quality_features(samples, timestamps=timestamps, sample_rate=sample_rate)


def quality_scores_for_rows(
    rows,
    *,
    sample_rate: float = 30.0,
    max_samples: int | None = None,
    log_component: str | None = None,
    log_label: str = "quality",
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    scores = []
    metadata: list[dict[str, float | str]] = []
    total = len(rows)
    if log_component is not None:
        log_event(log_component, "quality_start", label=log_label, total=total, max_samples=max_samples)
    progress_every = max(1, total // 10) if total else 1
    for index, row in enumerate(rows, start=1):
        features = compute_quality_from_jsonl(row.raw_path, sample_rate=sample_rate, max_samples=max_samples)
        score = float(features["quality_score"])
        scores.append(score)
        metadata_row: dict[str, float | str] = {
            "sample_id": row.sample_id,
            "split": row.split,
            "raw_path": str(row.raw_path),
        }
        metadata_row.update({key: float(value) for key, value in features.items()})
        metadata.append(metadata_row)
        if log_component is not None:
            log_progress(
                log_component,
                "quality_progress",
                index=index,
                total=total,
                every=progress_every,
                label=log_label,
                quality_score=score,
            )
    if log_component is not None:
        log_event(log_component, "quality_done", label=log_label, total=total)
    return np.asarray(scores, dtype=float), metadata


def load_modal_jsonl_imu(path: str | Path, *, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    raw_path = Path(path)
    samples: list[list[float]] = []
    timestamps: list[float] = []
    saw_timestamp = False
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and len(samples) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample = _sample_from_jsonl_record(record)
            samples.append(sample)
            timestamp = _timestamp_seconds_from_jsonl_record(record)
            if timestamp is None:
                timestamps.append(np.nan)
            else:
                timestamps.append(timestamp)
                saw_timestamp = True
    if not samples:
        raise ValueError(f"No IMU samples found in {raw_path}")
    timestamp_array = np.asarray(timestamps, dtype=float) if saw_timestamp else None
    return np.asarray(samples, dtype=float), timestamp_array


def score_quality(features: Mapping[str, float]) -> float:
    """Map artifact features to a conservative quality score in [0, 1]."""

    penalty = 0.0
    penalty += 1.40 * float(features.get("missing_rate", 0.0))
    penalty += 1.10 * float(features.get("nan_fraction", 0.0))
    penalty += 1.10 * float(features.get("inf_fraction", 0.0))
    penalty += 1.20 * float(features.get("flatline_fraction", 0.0))
    penalty += 1.30 * float(features.get("saturation_fraction", 0.0))
    penalty += 8.00 * min(float(features.get("extreme_value_fraction", 0.0)), 0.10)
    penalty += 5.00 * min(float(features.get("spike_rate", 0.0)), 0.25)
    penalty += 0.80 * float(features.get("repeated_timestamp_fraction", 0.0))
    penalty += 0.60 * float(features.get("timestamp_jitter_fraction", 0.0))

    axis_imbalance = float(features.get("axis_imbalance", 1.0))
    if axis_imbalance > 25.0:
        penalty += min((axis_imbalance - 25.0) / 100.0, 0.20)

    max_abs_value = float(features.get("max_abs_value", 0.0))
    if max_abs_value > 250.0:
        penalty += min((max_abs_value - 250.0) / 250.0, 0.75)

    stationary_fraction = float(features.get("stationary_fraction", 0.0))
    if stationary_fraction > 0.98:
        penalty += 0.15

    return float(np.clip(1.0 - penalty, 0.0, 1.0))


def _sample_from_jsonl_record(record: Mapping[str, object]) -> list[float]:
    if "acc" in record and "gyro" in record:
        acc = list(record["acc"])  # type: ignore[arg-type]
        gyro = list(record["gyro"])  # type: ignore[arg-type]
        if len(acc) < 3 or len(gyro) < 3:
            raise ValueError("JSONL records with acc/gyro must contain three values each")
        return [float(value) for value in [*acc[:3], *gyro[:3]]]

    if "gyro" in record:
        recovered = _recover_acc_with_mangled_key(record)
        if recovered is not None:
            acc, gyro = recovered
            return [float(value) for value in [*acc[:3], *gyro[:3]]]

    channel_keys = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    if all(key in record for key in channel_keys):
        return [float(record[key]) for key in channel_keys]  # type: ignore[arg-type]

    numeric = [
        float(value)
        for key, value in record.items()
        if key not in {"timestamp", "time", "t", "ts", "t_us"} and isinstance(value, int | float)
    ]
    if len(numeric) < 6:
        raise ValueError("JSONL IMU records must include acc/gyro arrays or at least six numeric channels")
    return numeric[:6]


def _recover_acc_with_mangled_key(record: Mapping[str, object]) -> tuple[list[float], list[float]] | None:
    gyro_value = record.get("gyro")
    if not isinstance(gyro_value, list | tuple) or len(gyro_value) < 3:
        return None
    vector_candidates = [
        value
        for key, value in record.items()
        if key not in {"gyro", "timestamp", "time", "t", "ts", "t_us"}
        and isinstance(value, list | tuple)
        and len(value) >= 3
    ]
    if len(vector_candidates) != 1:
        return None
    return list(vector_candidates[0]), list(gyro_value)


def _timestamp_seconds_from_jsonl_record(record: Mapping[str, object]) -> float | None:
    if "t_us" in record:
        return float(record["t_us"]) / 1_000_000.0  # type: ignore[arg-type]
    for key in ("timestamp", "time", "t", "ts"):
        if key in record:
            return float(record[key])  # type: ignore[arg-type]
    return None
