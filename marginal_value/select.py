from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd

from marginal_value.data.load_imu import load_worker_csv
from marginal_value.preprocessing.quality import compute_quality_features, load_modal_jsonl_imu
from marginal_value.preprocessing.window_features import compute_window_feature_matrix
from marginal_value.ranking.baseline_ranker import (
    annotate_cluster_features,
    compute_batch_clusters,
    old_knn_novelty,
    quality_gated_old_novelty_rank_rows,
    raw_shape_stats_embedding,
    temporal_order_embedding,
    window_mean_std_embedding,
)


DEFAULT_OUTPUT_COLUMNS = (
    "sample_id",
    "rank",
    "score",
    "quality_score",
    "old_novelty_score",
    "quality_gate_pass",
    "physical_validity_pass",
    "physical_validity_failure_reasons",
    "stationary_fraction",
    "max_abs_value",
    "new_cluster_id",
    "new_cluster_size",
    "was_selected_by_source_cap_fallback",
    "reranker",
    "raw_path",
)


def run_external_selector(
    *,
    old_support_path: str | Path,
    candidate_pool_path: str | Path,
    output_path: str | Path,
    sample_rate: float = 30.0,
    representation: str = "window_shape_stats",
    k_old: int = 5,
    quality_threshold: float = 0.85,
    max_stationary_fraction: float = 0.90,
    max_abs_value: float = 60.0,
    cluster_similarity_threshold: float = 0.985,
    source_cap: int | None = 2,
    max_samples: int | None = None,
    segment_old_support: bool = True,
    support_clip_seconds: float = 180.0,
    support_stride_seconds: float | None = None,
) -> dict[str, Any]:
    """Rank a candidate pool using only old support and candidate inputs.

    This is the external-evaluator interface. It does not accept a target set,
    labels, or evaluation representations.
    """

    support_rows = _load_manifest(old_support_path)
    candidate_rows = _load_manifest(candidate_pool_path)
    if not support_rows:
        raise ValueError("old support manifest must contain at least one row")
    if not candidate_rows:
        raise ValueError("candidate pool manifest must contain at least one row")

    support_embeddings = _support_embeddings(
        support_rows,
        representation=representation,
        sample_rate=sample_rate,
        max_samples=max_samples,
        segment_old_support=segment_old_support,
        support_clip_seconds=support_clip_seconds,
        support_stride_seconds=support_stride_seconds,
    )
    candidate_embeddings = _clip_embeddings(
        candidate_rows,
        representation=representation,
        sample_rate=sample_rate,
        max_samples=max_samples,
    )
    old_distances, _neighbors = old_knn_novelty(support_embeddings, candidate_embeddings, k=max(1, int(k_old)))
    quality_metadata = _quality_metadata(candidate_rows, sample_rate=sample_rate, max_samples=max_samples)

    scored = []
    for row, novelty, quality in zip(candidate_rows, old_distances, quality_metadata, strict=True):
        scored_row = {
            "sample_id": row["sample_id"],
            "raw_path": str(row["raw_path"]),
            "old_novelty_score": float(novelty),
            "quality_score": float(quality["quality_score"]),
            "stationary_fraction": float(quality["stationary_fraction"]),
            "max_abs_value": float(quality["max_abs_value"]),
        }
        if row.get("source_group_id"):
            scored_row["source_group_id"] = row["source_group_id"]
        scored.append(scored_row)

    cluster_ids = compute_batch_clusters(candidate_embeddings, similarity_threshold=float(cluster_similarity_threshold))
    scored = annotate_cluster_features(scored, candidate_embeddings, cluster_ids)
    ranked = quality_gated_old_novelty_rank_rows(
        scored,
        quality_threshold=float(quality_threshold),
        max_stationary_fraction=float(max_stationary_fraction),
        max_abs_value=float(max_abs_value),
        source_cap=source_cap,
        source_key="new_cluster_id",
    )
    output_rows = [_output_row(row) for row in ranked]
    _write_output(output_path, output_rows)
    return {
        "n_support": len(support_rows),
        "n_support_segments": int(len(support_embeddings)),
        "n_candidates": len(candidate_rows),
        "output_path": str(output_path),
        "selector": f"qgate_oldnovelty_{representation}",
        "representation": representation,
        "quality_threshold": float(quality_threshold),
        "max_stationary_fraction": float(max_stationary_fraction),
        "max_abs_value": float(max_abs_value),
        "source_cap": source_cap,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen external IMU candidate selector.")
    parser.add_argument("--old-support", required=True, help="CSV manifest for old support clips.")
    parser.add_argument("--candidate-pool", required=True, help="CSV manifest for candidate clips to rank.")
    parser.add_argument("--output", required=True, help="Output ranked candidate CSV.")
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument(
        "--representation",
        choices=["window_shape_stats", "window_mean_std_pool", "temporal_order", "raw_shape_stats"],
        default="window_shape_stats",
    )
    parser.add_argument("--k-old", type=int, default=5)
    parser.add_argument("--quality-threshold", type=float, default=0.85)
    parser.add_argument("--max-stationary-fraction", type=float, default=0.90)
    parser.add_argument("--max-abs-value", type=float, default=60.0)
    parser.add_argument("--cluster-similarity-threshold", type=float, default=0.985)
    parser.add_argument("--source-cap", type=int, default=2)
    parser.add_argument("--no-source-cap", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--support-clip-seconds", type=float, default=180.0)
    parser.add_argument("--support-stride-seconds", type=float)
    parser.add_argument("--no-segment-old-support", action="store_true")
    args = parser.parse_args(argv)
    source_cap = None if args.no_source_cap else args.source_cap
    result = run_external_selector(
        old_support_path=args.old_support,
        candidate_pool_path=args.candidate_pool,
        output_path=args.output,
        sample_rate=args.sample_rate,
        representation=args.representation,
        k_old=args.k_old,
        quality_threshold=args.quality_threshold,
        max_stationary_fraction=args.max_stationary_fraction,
        max_abs_value=args.max_abs_value,
        cluster_similarity_threshold=args.cluster_similarity_threshold,
        source_cap=source_cap,
        max_samples=args.max_samples,
        segment_old_support=not args.no_segment_old_support,
        support_clip_seconds=args.support_clip_seconds,
        support_stride_seconds=args.support_stride_seconds,
    )
    print(f"Wrote ranked candidates: {result['output_path']}")
    return 0


def _load_manifest(path: str | Path) -> list[dict[str, object]]:
    manifest_path = Path(path)
    if manifest_path.suffix.lower() != ".csv":
        return _load_line_manifest(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, record in enumerate(reader):
            raw_value = record.get("raw_path") or record.get("path") or record.get("file_path")
            if not raw_value:
                raise ValueError("manifest rows must include raw_path, path, or file_path")
            raw_path = _resolve_raw_reference(raw_value, manifest_path.parent)
            sample_id = (
                record.get("sample_id")
                or record.get("worker_id")
                or _sample_id_from_raw_reference(raw_path)
                or f"sample-{index}"
            )
            row: dict[str, object] = {
                "sample_id": str(sample_id),
                "raw_path": raw_path,
            }
            if record.get("source_group_id"):
                row["source_group_id"] = str(record["source_group_id"])
            rows.append(row)
    return rows


def _load_line_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw_value = line.strip()
        if not raw_value or raw_value.startswith("#"):
            continue
        raw_path = _resolve_raw_reference(raw_value, path.parent)
        rows.append(
            {
                "sample_id": _sample_id_from_raw_reference(raw_path) or f"sample-{index}",
                "raw_path": raw_path,
            }
        )
    return rows


def _resolve_raw_reference(value: str, base_dir: Path) -> str | Path:
    if _is_url(value):
        return value
    raw_path = Path(value)
    if not raw_path.is_absolute():
        raw_path = base_dir / raw_path
    return raw_path


def _sample_id_from_raw_reference(value: str | Path) -> str:
    if isinstance(value, Path):
        return value.stem
    parsed = urlparse(value)
    return Path(parsed.path).stem


def _is_url(value: str) -> bool:
    return urlparse(str(value)).scheme in {"http", "https", "file"}


def _support_embeddings(
    rows: Sequence[dict[str, object]],
    *,
    representation: str,
    sample_rate: float,
    max_samples: int | None,
    segment_old_support: bool,
    support_clip_seconds: float,
    support_stride_seconds: float | None,
) -> np.ndarray:
    embeddings = []
    for row in rows:
        samples, _timestamps = _load_imu_samples(row["raw_path"], max_samples=max_samples)
        if segment_old_support:
            for segment in _support_clip_segments(
                samples,
                sample_rate=sample_rate,
                clip_seconds=support_clip_seconds,
                stride_seconds=support_stride_seconds,
            ):
                embeddings.append(_clip_embedding(segment, representation=representation, sample_rate=sample_rate))
        else:
            embeddings.append(_clip_embedding(samples, representation=representation, sample_rate=sample_rate))
    return np.vstack(embeddings)


def _clip_embeddings(
    rows: Sequence[dict[str, object]],
    *,
    representation: str,
    sample_rate: float,
    max_samples: int | None,
) -> np.ndarray:
    embeddings = []
    for row in rows:
        samples, _timestamps = _load_imu_samples(row["raw_path"], max_samples=max_samples)
        embeddings.append(_clip_embedding(samples, representation=representation, sample_rate=sample_rate))
    return np.vstack(embeddings)


def _clip_embedding(samples: np.ndarray, *, representation: str, sample_rate: float) -> np.ndarray:
    if representation == "raw_shape_stats":
        return raw_shape_stats_embedding(samples, sample_rate=sample_rate)
    windows = compute_window_feature_matrix(samples)
    if representation == "window_shape_stats":
        return _window_shape_stats_embedding(windows)
    if representation == "window_mean_std_pool":
        return window_mean_std_embedding(windows)
    if representation == "temporal_order":
        return temporal_order_embedding(windows)
    raise ValueError(f"Unsupported selector representation: {representation}")


def _support_clip_segments(
    samples: np.ndarray,
    *,
    sample_rate: float,
    clip_seconds: float,
    stride_seconds: float | None,
) -> list[np.ndarray]:
    if clip_seconds <= 0.0:
        raise ValueError("support_clip_seconds must be positive.")
    if stride_seconds is not None and stride_seconds <= 0.0:
        raise ValueError("support_stride_seconds must be positive when provided.")
    values = np.asarray(samples, dtype=float)
    clip_samples = max(1, int(round(float(clip_seconds) * float(sample_rate))))
    stride_samples = max(1, int(round(float(stride_seconds or clip_seconds) * float(sample_rate))))
    if len(values) <= clip_samples:
        return [values]
    starts = list(range(0, len(values) - clip_samples + 1, stride_samples))
    if starts[-1] != len(values) - clip_samples:
        starts.append(len(values) - clip_samples)
    return [values[start : start + clip_samples] for start in starts]


def _window_shape_stats_embedding(windows: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(windows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("window_shape_stats requires a non-empty 2D window feature matrix.")
    if len(values) > 1:
        diffs = np.diff(values, axis=0)
        diff_parts = [np.mean(np.abs(diffs), axis=0), np.max(np.abs(diffs), axis=0)]
    else:
        diff_parts = [np.zeros(values.shape[1]), np.zeros(values.shape[1])]
    return np.nan_to_num(
        np.concatenate(
            [
                np.mean(values, axis=0),
                np.std(values, axis=0),
                np.min(values, axis=0),
                np.max(values, axis=0),
                np.percentile(values, 25, axis=0),
                np.percentile(values, 75, axis=0),
                *diff_parts,
            ]
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _quality_metadata(
    rows: Sequence[dict[str, object]],
    *,
    sample_rate: float,
    max_samples: int | None,
) -> list[dict[str, float]]:
    metadata = []
    for row in rows:
        samples, timestamps = _load_imu_samples(row["raw_path"], max_samples=max_samples)
        metadata.append(compute_quality_features(samples, timestamps=timestamps, sample_rate=sample_rate))
    return metadata


def _load_imu_samples(path: object, *, max_samples: int | None) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(path, str) and _is_url(path):
        return _load_jsonl_url(path, max_samples=max_samples)
    raw_path = Path(path)
    if raw_path.suffix.lower() == ".csv":
        worker = load_worker_csv(raw_path)
        samples = worker.samples[:max_samples] if max_samples is not None else worker.samples
        timestamps = worker.timestamps[:max_samples] if max_samples is not None and worker.timestamps is not None else worker.timestamps
        return samples, timestamps
    return load_modal_jsonl_imu(raw_path, max_samples=max_samples)


def _load_jsonl_url(url: str, *, max_samples: int | None) -> tuple[np.ndarray, np.ndarray | None]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    samples: list[list[float]] = []
    timestamps: list[float] = []
    saw_timestamp = False
    with urlopen(url) as response:  # nosec - user-supplied manifests intentionally identify input data.
        for raw_line in response:
            if max_samples is not None and len(samples) >= max_samples:
                break
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(_sample_from_jsonl_record(record))
            timestamp = _timestamp_seconds_from_jsonl_record(record)
            if timestamp is None:
                timestamps.append(np.nan)
            else:
                timestamps.append(timestamp)
                saw_timestamp = True
    if not samples:
        raise ValueError(f"No IMU samples found in {url}")
    timestamp_array = np.asarray(timestamps, dtype=float) if saw_timestamp else None
    return np.asarray(samples, dtype=float), timestamp_array


def _sample_from_jsonl_record(record: dict[str, object]) -> list[float]:
    if "acc" in record and "gyro" in record:
        acc = list(record["acc"])  # type: ignore[arg-type]
        gyro = list(record["gyro"])  # type: ignore[arg-type]
        if len(acc) < 3 or len(gyro) < 3:
            raise ValueError("JSONL records with acc/gyro must contain three values each")
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


def _timestamp_seconds_from_jsonl_record(record: dict[str, object]) -> float | None:
    if "t_us" in record:
        return float(record["t_us"]) / 1_000_000.0  # type: ignore[arg-type]
    for key in ("timestamp", "time", "t", "ts"):
        if key in record:
            return float(record[key])  # type: ignore[arg-type]
    return None


def _output_row(row: dict[str, object]) -> dict[str, object]:
    output = {column: row.get(column, "") for column in DEFAULT_OUTPUT_COLUMNS}
    output["score"] = float(row.get("rerank_score", row.get("quality_gate_old_novelty_score", 0.0)))
    output["rank"] = int(row["rank"])
    return output


def _write_output(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(DEFAULT_OUTPUT_COLUMNS)).to_csv(output_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
