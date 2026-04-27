from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

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
    compute_batch_clusters,
    old_knn_novelty,
    window_mean_std_embedding,
)
from marginal_value.ranking.modal_baseline_rank import (
    _build_raw_signal_corruption_eval,
    _join_grammar_features,
    _join_quality_metadata,
    _load_grammar_features,
    _maybe_apply_score_guards,
    _maybe_promote_grammar_scores,
    _maybe_split_large_clusters,
    _rank_rows,
)


def run_source_blocked_eval(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Run a harder old-corpus validation with source-group and feature-cluster blocks.

    Unlike the submission candidate eval, positives and negatives both come from
    the old physical-source corpus. Candidate rows are removed from the support
    index, and positive rows come from held-out source groups inside held-out
    feature clusters.
    """

    mode = "smoke" if smoke else "full"
    log_event("source_blocked_eval", "start", mode=mode)
    rng = np.random.default_rng(int(config.get("seed", 17)))
    data_root = Path(config["data"]["root"])
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    grammar_features, grammar_feature_path = _load_grammar_features(config, mode=mode)

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
    log_event("source_blocked_eval", "rows_ready", n_rows=len(rows), max_rows=max_rows)

    embeddings = _load_window_mean_std_embeddings(
        rows,
        workers=int(config["eval"].get("embedding_load_workers", 8)),
    )
    source_groups = np.asarray([_source_group_id(row.url) for row in rows], dtype=object)
    n_clusters = int(config["eval"].get("smoke_n_clusters" if smoke else "n_clusters", 64))
    source_cluster_ids = np.asarray(
        _deterministic_kmeans_labels(
            normalize_rows(embeddings),
            n_clusters=min(n_clusters, len(rows)),
            iterations=int(config["eval"].get("kmeans_iterations", 16)),
        ),
        dtype=int,
    )
    log_event(
        "source_blocked_eval",
        "source_clusters_done",
        **_cluster_summary(source_cluster_ids),
        n_source_groups=len(set(source_groups.tolist())),
    )

    folds = _select_source_blocked_folds(
        source_cluster_ids,
        source_groups,
        n_folds=int(config["eval"].get("smoke_folds" if smoke else "folds", 4)),
        clusters_per_fold=int(config["eval"].get("clusters_per_fold", 1)),
        source_groups_per_fold=int(config["eval"].get("source_groups_per_fold", 64)),
    )
    fold_reports: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        fold_report, candidate_rows = _run_source_blocked_fold(
            rows=rows,
            embeddings=embeddings,
            source_groups=source_groups,
            source_cluster_ids=source_cluster_ids,
            fold=fold,
            config=config,
            rng=rng,
            fold_index=fold_index,
            grammar_features=grammar_features,
        )
        fold_reports.append(fold_report)
        all_candidate_rows.extend(candidate_rows)
        log_event("source_blocked_eval", "fold_done", fold=fold_index, **fold_report["metrics"])

    report = {
        "mode": mode,
        "n_rows": len(rows),
        "n_source_groups": int(len(set(source_groups.tolist()))),
        "n_source_clusters": int(len(set(source_cluster_ids.tolist()))),
        "source_cluster_summary": _cluster_summary(source_cluster_ids),
        "folds": fold_reports,
        "mean_metrics": _mean_fold_metrics(fold_reports),
        "grammar_features": {
            "enabled": bool(config.get("grammar_features", {}).get("enabled", False)),
            "source_path": str(grammar_feature_path) if grammar_feature_path is not None else None,
            "n_rows": len(grammar_features),
        },
        "config": {
            "pretrain_manifest": config["data"]["pretrain_manifest"],
            "max_rows": max_rows,
            "n_clusters": n_clusters,
            "folds": len(folds),
        },
    }
    report_path = output_dir / f"source_blocked_eval_{mode}.json"
    candidates_path = output_dir / f"source_blocked_candidates_{mode}.jsonl"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in all_candidate_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    result = {
        "mode": mode,
        "n_rows": len(rows),
        "n_source_groups": report["n_source_groups"],
        "n_source_clusters": report["n_source_clusters"],
        "mean_ndcg@100": report["mean_metrics"].get("ndcg@100", report["mean_metrics"].get("ndcg@50", 0.0)),
        "mean_precision@100": report["mean_metrics"].get("precision@100", report["mean_metrics"].get("precision@50", 0.0)),
        "report_path": str(report_path),
        "candidates_path": str(candidates_path),
    }
    log_event("source_blocked_eval", "done", **result)
    return result


def _run_source_blocked_fold(
    *,
    rows: list[SplitSample],
    embeddings: np.ndarray,
    source_groups: np.ndarray,
    source_cluster_ids: np.ndarray,
    fold: dict[str, object],
    config: dict[str, Any],
    rng: np.random.Generator,
    fold_index: int,
    grammar_features: dict[str, dict[str, object]],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    heldout_clusters = np.asarray(fold["heldout_source_clusters"], dtype=int)
    heldout_groups = np.asarray(fold["heldout_source_groups"], dtype=object)
    heldout_cluster_mask = np.isin(source_cluster_ids, heldout_clusters)
    heldout_group_mask = np.isin(source_groups, heldout_groups)
    positive_pool = np.flatnonzero(heldout_cluster_mask & heldout_group_mask)
    novel_region_mask = heldout_cluster_mask | heldout_group_mask
    negative_pool = np.flatnonzero(~novel_region_mask)
    if len(positive_pool) == 0 or len(negative_pool) == 0:
        raise ValueError("Each source-blocked fold requires positive and source-covered negative rows.")

    max_positive = int(config["eval"].get("max_positive_per_fold", 500))
    max_negative = _negative_candidate_budget(negative_pool, config=config)
    positive_indices = _sample_indices(positive_pool, max_size=max_positive, rng=rng)
    negative_indices = _sample_indices(negative_pool, max_size=max_negative, rng=rng)
    clean_candidate_indices = np.concatenate([positive_indices, negative_indices])
    support_mask = ~novel_region_mask
    support_mask[clean_candidate_indices] = False
    support_indices = np.flatnonzero(support_mask)
    if len(support_indices) == 0:
        raise ValueError("Each source-blocked fold requires non-candidate source-covered support rows.")

    clean_rows = [rows[int(idx)] for idx in clean_candidate_indices]
    clean_embeddings = embeddings[clean_candidate_indices]
    clean_labels = np.concatenate([np.ones(len(positive_indices), dtype=int), np.zeros(len(negative_indices), dtype=int)])
    clean_types = ["heldout_source_cluster_positive"] * len(positive_indices) + ["source_covered_hard_negative"] * len(negative_indices)

    corruption_eval = _maybe_build_fold_corruptions(
        positive_rows=[rows[int(idx)] for idx in positive_indices],
        config=config,
        rng=rng,
        sample_rate=float(config.get("quality", {}).get("sample_rate", 30.0)),
        max_samples=config.get("quality", {}).get("max_samples_per_clip"),
    )
    if len(corruption_eval["embeddings"]):
        candidate_embeddings = np.vstack([clean_embeddings, corruption_eval["embeddings"]])
    else:
        candidate_embeddings = clean_embeddings
    labels = np.concatenate([clean_labels, corruption_eval["labels"]]) if len(corruption_eval["labels"]) else clean_labels
    is_corruption = np.concatenate(
        [np.zeros(len(clean_labels), dtype=bool), corruption_eval["is_corruption"]]
    )
    sample_ids = [row.sample_id for row in clean_rows] + list(corruption_eval["sample_ids"])
    source_sample_ids = [""] * len(clean_rows) + list(corruption_eval["source_sample_ids"])
    candidate_types = clean_types + ["raw_corruption_negative"] * len(corruption_eval["sample_ids"])
    sample_source_groups = [str(source_groups[int(idx)]) for idx in clean_candidate_indices] + ["corruption"] * len(corruption_eval["sample_ids"])
    sample_source_clusters = [int(source_cluster_ids[int(idx)]) for idx in clean_candidate_indices] + [-1] * len(corruption_eval["sample_ids"])

    support_embeddings = embeddings[support_indices]
    old_distance, _neighbors = old_knn_novelty(
        support_embeddings,
        candidate_embeddings,
        k=int(config["eval"].get("k_old", 10)),
    )
    density = batch_density(candidate_embeddings, k=int(config["eval"].get("k_new_density", 10)))
    clean_quality, clean_quality_metadata = quality_scores_for_rows(
        clean_rows,
        sample_rate=float(config.get("quality", {}).get("sample_rate", 30.0)),
        max_samples=config.get("quality", {}).get("max_samples_per_clip"),
        log_component="source_blocked_eval",
        log_label=f"fold_{fold_index}",
    )
    quality_scores = np.concatenate([clean_quality, corruption_eval["quality_scores"]])
    scored = build_scored_rows(
        sample_ids=sample_ids,
        embeddings=candidate_embeddings,
        old_knn_distance=old_distance,
        new_density=density,
        quality_scores=quality_scores,
        novelty_weight=float(config["eval"].get("novelty_weight", config.get("ranking", {}).get("novelty_weight", 0.75))),
    )
    quality_metadata = clean_quality_metadata + corruption_eval["quality_metadata"]
    scored = _join_quality_metadata(scored, quality_metadata)
    for idx, row in enumerate(scored):
        row["candidate_index"] = int(idx)
        row["fold"] = int(fold_index)
        row["label"] = int(labels[idx])
        row["is_corruption"] = bool(is_corruption[idx])
        row["candidate_type"] = candidate_types[idx]
        row["source_group_id"] = sample_source_groups[idx]
        row["source_cluster_id"] = int(sample_source_clusters[idx])
        row["source_sample_id"] = source_sample_ids[idx]
        row["heldout_source_groups"] = ",".join(str(group) for group in heldout_groups.tolist())
        row["heldout_source_clusters"] = ",".join(str(cluster) for cluster in heldout_clusters.tolist())

    scored = _join_grammar_features(scored, grammar_features)
    scored, grammar_summary = _maybe_promote_grammar_scores(config, scored, label=f"source_blocked_fold_{fold_index}")
    candidate_cluster_ids = compute_batch_clusters(
        candidate_embeddings,
        similarity_threshold=float(config["eval"].get("candidate_cluster_similarity_threshold", config.get("ranking", {}).get("cluster_similarity_threshold", 0.985))),
    )
    scored = annotate_cluster_features(scored, candidate_embeddings, candidate_cluster_ids)
    scored, score_guards = _maybe_apply_score_guards(config, scored, label=f"source_blocked_fold_{fold_index}")
    scored, large_cluster_split = _maybe_split_large_clusters(
        config,
        scored,
        candidate_embeddings,
        label=f"source_blocked_fold_{fold_index}",
    )
    ranking_config = config.get("ranking", {})
    ranked = _rank_rows(
        scored,
        candidate_embeddings,
        reranker_method=str(ranking_config.get("reranker_method", "cluster_cap")),
        mmr_lambda=float(ranking_config.get("mmr_lambda", config["eval"].get("mmr_lambda", 0.0))),
        cluster_bonus_weight=float(ranking_config.get("cluster_bonus_weight", config["eval"].get("cluster_bonus_weight", 0.0))),
        cluster_cap_top_k=int(ranking_config.get("cluster_cap_top_k", config["eval"].get("cluster_cap_top_k", 200))),
        cluster_max_per_cluster=int(ranking_config.get("cluster_max_per_cluster", config["eval"].get("cluster_max_per_cluster", 8))),
        cluster_cap_key=str(ranking_config.get("cluster_cap_key", config["eval"].get("cluster_cap_key", "new_cluster_id"))),
        cluster_cap_min_quality=float(ranking_config.get("cluster_cap_min_quality", config["eval"].get("cluster_cap_min_quality", 0.45))),
        cluster_cap_schedule=ranking_config.get("cluster_cap_schedule"),
        prefix_cluster_cap_top_k=int(ranking_config.get("prefix_cluster_cap_top_k", config["eval"].get("prefix_cluster_cap_top_k", 75))),
        prefix_cluster_cap_key=str(ranking_config.get("prefix_cluster_cap_key", config["eval"].get("prefix_cluster_cap_key", "new_cluster_parent_id"))),
        prefix_cluster_max_per_cluster=int(ranking_config.get("prefix_cluster_max_per_cluster", ranking_config.get("cluster_max_per_cluster", 8))),
    )
    order = np.asarray([int(row["candidate_index"]) for row in ranked], dtype=int)
    ranked_source_clusters = np.asarray([int(scored[idx].get("source_cluster_id", -1)) for idx in order], dtype=int)
    metrics = summarize_ranked_scores(
        scores=np.arange(len(order), 0, -1, dtype=float),
        labels=labels[order],
        clusters=ranked_source_clusters,
        corruptions=is_corruption[order],
        k_values=[int(k) for k in config["eval"].get("k_values", [10, 50, 100, 200])],
    )
    metrics["mean_quality"] = float(np.mean(quality_scores)) if len(quality_scores) else 0.0
    metrics["low_quality_fraction"] = float(np.mean(quality_scores < float(config["eval"].get("low_quality_threshold", 0.45)))) if len(quality_scores) else 0.0

    report = {
        "fold": int(fold_index),
        "heldout_source_clusters": [int(cluster) for cluster in heldout_clusters.tolist()],
        "heldout_source_groups": [str(group) for group in heldout_groups.tolist()],
        "n_support": int(len(support_indices)),
        "n_positive": int(len(positive_indices)),
        "n_negative": int(len(negative_indices)),
        "n_corruption": int(len(corruption_eval["sample_ids"])),
        "metrics": metrics,
        "top_k": _top_k_report(ranked, labels=labels, corruptions=is_corruption, k_values=config["eval"].get("k_values", [10, 50, 100, 200])),
        "grammar_promotion": grammar_summary,
        "score_guards": score_guards,
        "large_cluster_split": large_cluster_split,
        "corruption_eval": corruption_eval["summary"],
    }
    return report, ranked


def _maybe_build_fold_corruptions(
    *,
    positive_rows: list[SplitSample],
    config: dict[str, Any],
    rng: np.random.Generator,
    sample_rate: float,
    max_samples: int | None,
) -> dict[str, Any]:
    corruption_config = config.get("corruption_eval", {})
    if not isinstance(corruption_config, dict) or not bool(corruption_config.get("enabled", False)):
        return _empty_corruption_eval("disabled")
    if not bool(corruption_config.get("raw_signal", True)):
        raise ValueError("source-blocked eval requires corruption_eval.raw_signal=true when corruption eval is enabled.")
    if not positive_rows:
        return _empty_corruption_eval("no_positive_rows")
    sample_size = min(int(corruption_config.get("sample_size", 64)), len(positive_rows))
    if sample_size <= 0:
        return _empty_corruption_eval("sample_size_zero")
    selected = rng.choice(len(positive_rows), size=sample_size, replace=False)
    return _build_raw_signal_corruption_eval(
        positive_rows=positive_rows,
        selected_indices=selected,
        modes=[str(mode) for mode in corruption_config.get("modes", ["flatline", "spike", "saturation", "jitter"])],
        rng=rng,
        sample_rate=sample_rate,
        max_samples=max_samples,
    )


def _negative_candidate_budget(negative_pool: np.ndarray, *, config: dict[str, Any]) -> int:
    configured_max = int(config["eval"].get("max_negative_per_fold", 500))
    if len(negative_pool) <= 1:
        return configured_max
    default_min_support = max(1, min(100, len(negative_pool) // 2))
    min_support = int(config["eval"].get("min_source_covered_support_per_fold", default_min_support))
    min_support = max(1, min(min_support, len(negative_pool) - 1))
    return max(1, min(configured_max, len(negative_pool) - min_support))


def _empty_corruption_eval(reason: str) -> dict[str, Any]:
    return {
        "embeddings": np.empty((0, 0), dtype=float),
        "labels": np.asarray([], dtype=int),
        "is_corruption": np.asarray([], dtype=bool),
        "quality_scores": np.asarray([], dtype=float),
        "sample_ids": [],
        "source_sample_ids": [],
        "splits": [],
        "modes": [],
        "quality_metadata": [],
        "summary": {"enabled": False, "n_corruptions": 0, "skip_reason": reason},
    }


def _load_window_mean_std_embeddings(rows: list[SplitSample], *, workers: int) -> np.ndarray:
    total = len(rows)
    if total == 0:
        raise ValueError("No rows available for source-blocked eval.")
    workers = max(1, int(workers))
    log_event("source_blocked_eval", "embedding_load_start", total=total, workers=workers)
    progress_every = max(1, total // 20)
    embeddings: list[np.ndarray | None] = [None] * total
    if workers == 1:
        for idx, row in enumerate(rows, start=1):
            embeddings[idx - 1] = _load_one_embedding(row)
            log_progress("source_blocked_eval", "embedding_load_progress", index=idx, total=total, every=progress_every)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = {executor.submit(_load_one_embedding, row): idx for idx, row in enumerate(rows)}
            for completed, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                embeddings[idx] = future.result()
                log_progress("source_blocked_eval", "embedding_load_progress", index=completed, total=total, every=progress_every)
    loaded = [embedding for embedding in embeddings if embedding is not None]
    if len(loaded) != total:
        raise RuntimeError(f"Loaded {len(loaded)} embeddings for {total} rows.")
    log_event("source_blocked_eval", "embedding_load_done", total=total)
    return np.vstack(loaded)


def _load_one_embedding(row: SplitSample) -> np.ndarray:
    with np.load(row.feature_path) as data:
        return window_mean_std_embedding(np.asarray(data["window_features"], dtype="float32"))


def _select_source_blocked_folds(
    cluster_ids: np.ndarray,
    source_groups: np.ndarray,
    *,
    n_folds: int,
    clusters_per_fold: int,
    source_groups_per_fold: int,
) -> list[dict[str, object]]:
    unique, counts = np.unique(cluster_ids, return_counts=True)
    ordered_clusters = unique[np.argsort(-counts)]
    folds: list[dict[str, object]] = []
    cursor = 0
    for _fold in range(max(1, n_folds)):
        heldout_clusters = [int(cluster_id) for cluster_id in ordered_clusters[cursor : cursor + clusters_per_fold]]
        if not heldout_clusters:
            break
        mask = np.isin(cluster_ids, np.asarray(heldout_clusters, dtype=int))
        group_counts = Counter(str(group) for group in source_groups[mask].tolist())
        groups = [group for group, _count in sorted(group_counts.items(), key=lambda item: (-item[1], item[0]))]
        if source_groups_per_fold > 0:
            groups = groups[:source_groups_per_fold]
        if not groups:
            cursor += clusters_per_fold
            continue
        folds.append({"heldout_source_clusters": heldout_clusters, "heldout_source_groups": groups})
        cursor += clusters_per_fold
    return folds


def _top_k_report(
    ranked_rows: list[dict[str, object]],
    *,
    labels: np.ndarray,
    corruptions: np.ndarray,
    k_values: Iterable[int],
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    order = [int(row.get("candidate_index", 0)) for row in ranked_rows]
    for k in k_values:
        subset = ranked_rows[: int(k)]
        subset_order = order[: int(k)]
        subset_labels = [int(labels[idx]) for idx in subset_order]
        subset_corruptions = [bool(corruptions[idx]) for idx in subset_order]
        source_groups = [str(row.get("source_group_id", "")) for row in subset]
        source_clusters = [str(row.get("source_cluster_id", "")) for row in subset]
        qualities = [float(row.get("quality_score", 0.0)) for row in subset]
        report[str(k)] = {
            "n_rows": len(subset),
            "positive_fraction": _mean(subset_labels),
            "corruption_fraction": _mean([1.0 if value else 0.0 for value in subset_corruptions]),
            "mean_quality": _mean(qualities),
            "low_quality_count": int(sum(value < 0.45 for value in qualities)),
            "unique_source_groups": int(len(set(source_groups))),
            "unique_source_clusters": int(len(set(source_clusters))),
            "largest_source_group_fraction": _largest_fraction(source_groups),
            "largest_source_cluster_fraction": _largest_fraction(source_clusters),
            "candidate_type_counts": dict(Counter(str(row.get("candidate_type", "")) for row in subset)),
        }
    return report


def _source_group_id(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
    for part in parts:
        if part.startswith("worker"):
            return part
    return parsed.netloc + ":" + PurePosixPath(parsed.path).name


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


def _largest_fraction(values: list[str]) -> float:
    if not values:
        return 0.0
    return float(Counter(values).most_common(1)[0][1] / len(values))


def _mean(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    return float(sum(numbers) / len(numbers)) if numbers else 0.0
