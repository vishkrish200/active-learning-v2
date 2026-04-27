from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from marginal_value.data.split_manifest import SplitSample, build_split_manifest, select_split
from marginal_value.eval.ablation_eval import summarize_ranked_scores
from marginal_value.indexing.knn_features import normalize_rows
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import quality_scores_for_rows
from marginal_value.ranking.baseline_ranker import (
    _deterministic_kmeans_labels,
    annotate_cluster_features,
    batch_density,
    build_scored_rows,
    cluster_cap_rank_rows,
    compute_batch_clusters,
    old_knn_novelty,
    window_mean_std_embedding,
)


def run_physical_leave_cluster_eval(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Evaluate old-support novelty by withholding clusters from physical pretrain data."""

    log_event("physical_lco", "start", smoke=smoke)
    rng = np.random.default_rng(int(config.get("seed", 17)))
    data_root = Path(config["data"]["root"])
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if smoke else "full"

    manifest = build_split_manifest(
        data_root,
        pretrain_manifest=config["data"]["pretrain_manifest"],
        val_manifest=config["data"]["val_manifest"],
        extra_manifests={},
        feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
        raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
    )
    rows = select_split(manifest, "pretrain")
    max_rows = int(config["eval"].get("smoke_max_rows" if smoke else "max_rows", len(rows)))
    rows = _sample_rows(rows, max_rows=max_rows, rng=rng)
    log_event("physical_lco", "rows_ready", n_rows=len(rows), max_rows=max_rows)

    embeddings = _load_window_mean_std_embeddings(
        rows,
        workers=int(config["eval"].get("embedding_load_workers", 8)),
    )
    cluster_count = int(config["eval"].get("smoke_n_clusters" if smoke else "n_clusters", 64))
    source_cluster_ids = np.asarray(
        _deterministic_kmeans_labels(
            normalize_rows(embeddings),
            n_clusters=min(cluster_count, len(rows)),
            iterations=int(config["eval"].get("kmeans_iterations", 16)),
        ),
        dtype=int,
    )
    log_event("physical_lco", "source_clusters_done", **_cluster_summary(source_cluster_ids))

    folds = _select_holdout_clusters(
        source_cluster_ids,
        n_folds=int(config["eval"].get("smoke_folds" if smoke else "folds", 4)),
        clusters_per_fold=int(config["eval"].get("clusters_per_fold", 1)),
    )
    fold_reports = []
    all_candidate_rows = []
    for fold_index, holdout_clusters in enumerate(folds):
        fold_report, candidate_rows = _run_fold(
            rows=rows,
            embeddings=embeddings,
            source_cluster_ids=source_cluster_ids,
            holdout_clusters=holdout_clusters,
            config=config,
            rng=rng,
            fold_index=fold_index,
        )
        fold_reports.append(fold_report)
        all_candidate_rows.extend(candidate_rows)
        log_event("physical_lco", "fold_done", fold=fold_index, **fold_report["metrics"])

    report = {
        "mode": suffix,
        "n_rows": len(rows),
        "n_source_clusters": int(len(set(source_cluster_ids.tolist()))),
        "source_cluster_summary": _cluster_summary(source_cluster_ids),
        "folds": fold_reports,
        "mean_metrics": _mean_fold_metrics(fold_reports),
        "config": {
            "pretrain_manifest": config["data"]["pretrain_manifest"],
            "max_rows": max_rows,
            "n_clusters": cluster_count,
            "folds": len(folds),
        },
    }
    report_path = output_dir / f"physical_leave_cluster_eval_{suffix}.json"
    candidates_path = output_dir / f"physical_leave_cluster_candidates_{suffix}.jsonl"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in all_candidate_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    result = {
        "mode": suffix,
        "n_rows": len(rows),
        "n_source_clusters": report["n_source_clusters"],
        "mean_ndcg@100": report["mean_metrics"].get("ndcg@100", report["mean_metrics"].get("ndcg@50", 0.0)),
        "mean_precision@100": report["mean_metrics"].get("precision@100", report["mean_metrics"].get("precision@50", 0.0)),
        "report_path": str(report_path),
        "candidates_path": str(candidates_path),
    }
    log_event("physical_lco", "done", **result)
    return result


def _run_fold(
    *,
    rows: list[SplitSample],
    embeddings: np.ndarray,
    source_cluster_ids: np.ndarray,
    holdout_clusters: list[int],
    config: dict[str, Any],
    rng: np.random.Generator,
    fold_index: int,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    holdout_mask = np.isin(source_cluster_ids, np.asarray(holdout_clusters, dtype=int))
    support_indices = np.flatnonzero(~holdout_mask)
    positive_indices = np.flatnonzero(holdout_mask)
    if len(support_indices) == 0 or len(positive_indices) == 0:
        raise ValueError("Each leave-cluster fold requires support and positive rows.")

    max_positive = int(config["eval"].get("max_positive_per_fold", 500))
    max_negative = int(config["eval"].get("max_negative_per_fold", 500))
    positive_indices = _sample_indices(positive_indices, max_size=max_positive, rng=rng)
    negative_indices = _sample_indices(support_indices, max_size=max_negative, rng=rng)
    candidate_indices = np.concatenate([positive_indices, negative_indices])
    labels = np.concatenate([np.ones(len(positive_indices), dtype=int), np.zeros(len(negative_indices), dtype=int)])

    support_embeddings = embeddings[support_indices]
    candidate_embeddings = embeddings[candidate_indices]
    candidate_rows = [rows[int(idx)] for idx in candidate_indices]
    old_distance, _neighbors = old_knn_novelty(
        support_embeddings,
        candidate_embeddings,
        k=int(config["eval"].get("k_old", 10)),
    )
    density = batch_density(candidate_embeddings, k=int(config["eval"].get("k_new_density", 10)))
    quality, quality_metadata = quality_scores_for_rows(
        candidate_rows,
        sample_rate=float(config.get("quality", {}).get("sample_rate", 30.0)),
        max_samples=config.get("quality", {}).get("max_samples_per_clip"),
        log_component="physical_lco",
        log_label=f"fold_{fold_index}",
    )
    scored = build_scored_rows(
        sample_ids=[row.sample_id for row in candidate_rows],
        embeddings=candidate_embeddings,
        old_knn_distance=old_distance,
        new_density=density,
        quality_scores=quality,
        novelty_weight=float(config["eval"].get("novelty_weight", 0.75)),
    )
    candidate_cluster_ids = compute_batch_clusters(
        candidate_embeddings,
        similarity_threshold=float(config["eval"].get("candidate_cluster_similarity_threshold", 0.985)),
    )
    scored = annotate_cluster_features(scored, candidate_embeddings, candidate_cluster_ids)
    for row, label, source_cluster_id in zip(scored, labels, source_cluster_ids[candidate_indices], strict=True):
        row["label"] = int(label)
        row["source_cluster_id"] = int(source_cluster_id)
        row["fold"] = int(fold_index)
        row["heldout_source_clusters"] = ",".join(str(cluster_id) for cluster_id in holdout_clusters)
        row["is_corruption"] = False
    ranked = cluster_cap_rank_rows(
        scored,
        candidate_embeddings,
        lambda_redundancy=float(config["eval"].get("mmr_lambda", 0.0)),
        cluster_bonus_weight=float(config["eval"].get("cluster_bonus_weight", 0.0)),
        cluster_cap_top_k=int(config["eval"].get("cluster_cap_top_k", 200)),
        cluster_max_per_cluster=int(config["eval"].get("cluster_max_per_cluster", 8)),
        cluster_key=str(config["eval"].get("cluster_cap_key", "new_cluster_id")),
        cluster_cap_min_quality=float(config["eval"].get("cluster_cap_min_quality", 0.45)),
    )
    order = np.asarray([int(row["sample_id"] in {rows[int(idx)].sample_id for idx in positive_indices}) for row in ranked], dtype=int)
    ranked_source_clusters = np.asarray([int(row["source_cluster_id"]) for row in ranked], dtype=int)
    metrics = summarize_ranked_scores(
        scores=np.arange(len(ranked), 0, -1, dtype=float),
        labels=order,
        clusters=ranked_source_clusters,
        corruptions=np.zeros(len(ranked), dtype=bool),
        k_values=[int(k) for k in config["eval"].get("k_values", [10, 50, 100, 200])],
    )
    metrics["mean_quality"] = float(np.mean(quality)) if len(quality) else 0.0
    metrics["low_quality_fraction"] = float(np.mean(quality < 0.45)) if len(quality) else 0.0
    report = {
        "fold": int(fold_index),
        "holdout_clusters": holdout_clusters,
        "n_support": int(len(support_indices)),
        "n_positive": int(len(positive_indices)),
        "n_negative": int(len(negative_indices)),
        "quality_metadata_rows": len(quality_metadata),
        "metrics": metrics,
    }
    return report, ranked


def _load_window_mean_std_embeddings(rows: list[SplitSample], *, workers: int) -> np.ndarray:
    total = len(rows)
    if total == 0:
        raise ValueError("No rows available for physical leave-cluster eval.")
    workers = max(1, int(workers))
    log_event("physical_lco", "embedding_load_start", total=total, workers=workers)
    progress_every = max(1, total // 20)
    embeddings: list[np.ndarray | None] = [None] * total
    if workers == 1:
        for idx, row in enumerate(rows, start=1):
            embeddings[idx - 1] = _load_one_embedding(row)
            log_progress("physical_lco", "embedding_load_progress", index=idx, total=total, every=progress_every)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = {executor.submit(_load_one_embedding, row): idx for idx, row in enumerate(rows)}
            for completed, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                embeddings[idx] = future.result()
                log_progress("physical_lco", "embedding_load_progress", index=completed, total=total, every=progress_every)
    loaded = [embedding for embedding in embeddings if embedding is not None]
    if len(loaded) != total:
        raise RuntimeError(f"Loaded {len(loaded)} embeddings for {total} rows.")
    log_event("physical_lco", "embedding_load_done", total=total)
    return np.vstack(loaded)


def _load_one_embedding(row: SplitSample) -> np.ndarray:
    with np.load(row.feature_path) as data:
        return window_mean_std_embedding(np.asarray(data["window_features"], dtype="float32"))


def _sample_rows(rows: list[SplitSample], *, max_rows: int, rng: np.random.Generator) -> list[SplitSample]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    indices = rng.choice(len(rows), size=max_rows, replace=False)
    return [rows[int(idx)] for idx in np.sort(indices)]


def _sample_indices(indices: np.ndarray, *, max_size: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if max_size <= 0 or len(values) <= max_size:
        return values
    sampled = rng.choice(values, size=max_size, replace=False)
    return np.asarray(sampled, dtype=int)


def _select_holdout_clusters(cluster_ids: np.ndarray, *, n_folds: int, clusters_per_fold: int) -> list[list[int]]:
    unique, counts = np.unique(cluster_ids, return_counts=True)
    order = unique[np.argsort(-counts)]
    folds = []
    cursor = 0
    for _fold in range(max(1, n_folds)):
        selected = [int(cluster_id) for cluster_id in order[cursor : cursor + clusters_per_fold]]
        if not selected:
            break
        folds.append(selected)
        cursor += clusters_per_fold
    return folds


def _mean_fold_metrics(fold_reports: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for report in fold_reports:
        for key, value in report["metrics"].items():
            values.setdefault(key, []).append(float(value))
    return {key: float(np.mean(metric_values)) for key, metric_values in values.items()}


def _cluster_summary(cluster_ids: np.ndarray) -> dict[str, float]:
    ids = np.asarray(cluster_ids, dtype=int)
    if ids.size == 0:
        return {"n_clusters": 0.0, "max_cluster_size": 0.0, "mean_cluster_size": 0.0}
    _unique, counts = np.unique(ids, return_counts=True)
    return {
        "n_clusters": float(len(counts)),
        "max_cluster_size": float(np.max(counts)),
        "mean_cluster_size": float(np.mean(counts)),
    }
