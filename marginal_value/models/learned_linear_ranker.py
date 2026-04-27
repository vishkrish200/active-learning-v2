from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LinearRankerModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float


def fit_linear_ranker(values: np.ndarray, labels: np.ndarray) -> LinearRankerModel:
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2:
        raise ValueError("values must be a 2D matrix.")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("fit_linear_ranker requires both positive and negative labels.")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1.0e-12, 1.0, scale)
    z = (x - mean) / scale
    positive_mean = np.mean(z[y == 1], axis=0)
    negative_mean = np.mean(z[y == 0], axis=0)
    weights = positive_mean - negative_mean
    midpoint = 0.5 * (positive_mean + negative_mean)
    bias = -float(midpoint @ weights)
    return LinearRankerModel(feature_mean=mean, feature_scale=scale, weights=weights, bias=bias)


def score_linear_ranker(model: LinearRankerModel, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2:
        raise ValueError("values must be a 2D matrix.")
    if x.shape[1] != len(model.weights):
        raise ValueError(f"Expected {len(model.weights)} model features, got {x.shape[1]}.")
    z = (x - model.feature_mean) / model.feature_scale
    return z @ model.weights + model.bias


def sigmoid_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def feature_matrix_from_rows(rows: Sequence[dict[str, object]], feature_names: Iterable[str]) -> np.ndarray:
    names = list(feature_names)
    columns = []
    missing: dict[str, int] = {}
    nonfinite: dict[str, int] = {}
    for name in names:
        raw_values = [row.get(name, np.nan) for row in rows]
        values = np.asarray([_to_float(value) for value in raw_values], dtype=float)
        missing_count = int(sum(name not in row for row in rows))
        nonfinite_count = int(np.sum(~np.isfinite(values)))
        if missing_count:
            missing[name] = missing_count
        if nonfinite_count:
            nonfinite[name] = nonfinite_count
        columns.append(values)
    if missing or nonfinite:
        details = {"missing": missing, "nonfinite": nonfinite}
        raise ValueError(f"Rows are missing required learned-ranker features: {details}")
    if not columns:
        raise ValueError("At least one learned-ranker feature is required.")
    return np.column_stack(columns)


def write_linear_ranker_model(
    path: str | Path,
    model: LinearRankerModel,
    *,
    feature_names: Sequence[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_type": "linear_centroid",
        "feature_names": list(feature_names),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "weights": model.weights.tolist(),
        "bias": float(model.bias),
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_linear_ranker_model(path: str | Path) -> tuple[LinearRankerModel, list[str], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("model_type") != "linear_centroid":
        raise ValueError(f"Unsupported learned-ranker model type: {payload.get('model_type')}")
    feature_names = [str(value) for value in payload.get("feature_names", [])]
    if not feature_names:
        raise ValueError("Learned-ranker model has no feature_names.")
    model = LinearRankerModel(
        feature_mean=np.asarray(payload["feature_mean"], dtype=float),
        feature_scale=np.asarray(payload["feature_scale"], dtype=float),
        weights=np.asarray(payload["weights"], dtype=float),
        bias=float(payload["bias"]),
    )
    if len(feature_names) != len(model.weights):
        raise ValueError("Learned-ranker model feature_names and weights have different lengths.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return model, feature_names, metadata


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return 1.0 if value.strip().lower() == "true" else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
