from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from marginal_value.data.load_imu import load_worker_csv
from marginal_value.preprocessing.quality import compute_quality_features, load_modal_jsonl_imu
from marginal_value.ranking.baseline_ranker import (
    annotate_cluster_features,
    compute_batch_clusters,
    old_knn_novelty,
    quality_gated_old_novelty_rank_rows,
    raw_shape_stats_embedding,
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
    k_old: int = 5,
    quality_threshold: float = 0.85,
    max_stationary_fraction: float = 0.90,
    max_abs_value: float = 60.0,
    cluster_similarity_threshold: float = 0.985,
    source_cap: int | None = 2,
    max_samples: int | None = None,
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

    support_embeddings = _raw_shape_embeddings(support_rows, sample_rate=sample_rate, max_samples=max_samples)
    candidate_embeddings = _raw_shape_embeddings(candidate_rows, sample_rate=sample_rate, max_samples=max_samples)
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
        "n_candidates": len(candidate_rows),
        "output_path": str(output_path),
        "selector": "qgate_oldnovelty_raw_shape_stats",
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
    parser.add_argument("--k-old", type=int, default=5)
    parser.add_argument("--quality-threshold", type=float, default=0.85)
    parser.add_argument("--max-stationary-fraction", type=float, default=0.90)
    parser.add_argument("--max-abs-value", type=float, default=60.0)
    parser.add_argument("--cluster-similarity-threshold", type=float, default=0.985)
    parser.add_argument("--source-cap", type=int, default=2)
    parser.add_argument("--no-source-cap", action="store_true")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    source_cap = None if args.no_source_cap else args.source_cap
    result = run_external_selector(
        old_support_path=args.old_support,
        candidate_pool_path=args.candidate_pool,
        output_path=args.output,
        sample_rate=args.sample_rate,
        k_old=args.k_old,
        quality_threshold=args.quality_threshold,
        max_stationary_fraction=args.max_stationary_fraction,
        max_abs_value=args.max_abs_value,
        cluster_similarity_threshold=args.cluster_similarity_threshold,
        source_cap=source_cap,
        max_samples=args.max_samples,
    )
    print(f"Wrote ranked candidates: {result['output_path']}")
    return 0


def _load_manifest(path: str | Path) -> list[dict[str, object]]:
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, record in enumerate(reader):
            raw_value = record.get("raw_path") or record.get("path") or record.get("file_path")
            if not raw_value:
                raise ValueError("manifest rows must include raw_path, path, or file_path")
            raw_path = Path(raw_value)
            if not raw_path.is_absolute():
                raw_path = manifest_path.parent / raw_path
            sample_id = record.get("sample_id") or record.get("worker_id") or raw_path.stem or f"sample-{index}"
            row: dict[str, object] = {
                "sample_id": str(sample_id),
                "raw_path": raw_path,
            }
            if record.get("source_group_id"):
                row["source_group_id"] = str(record["source_group_id"])
            rows.append(row)
    return rows


def _raw_shape_embeddings(
    rows: Sequence[dict[str, object]],
    *,
    sample_rate: float,
    max_samples: int | None,
) -> np.ndarray:
    embeddings = []
    for row in rows:
        samples, _timestamps = _load_imu_samples(row["raw_path"], max_samples=max_samples)
        embeddings.append(raw_shape_stats_embedding(samples, sample_rate=sample_rate))
    return np.vstack(embeddings)


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
    raw_path = Path(path)
    if raw_path.suffix.lower() == ".csv":
        worker = load_worker_csv(raw_path)
        samples = worker.samples[:max_samples] if max_samples is not None else worker.samples
        timestamps = worker.timestamps[:max_samples] if max_samples is not None and worker.timestamps is not None else worker.timestamps
        return samples, timestamps
    return load_modal_jsonl_imu(raw_path, max_samples=max_samples)


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
