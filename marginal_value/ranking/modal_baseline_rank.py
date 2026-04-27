from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np

from marginal_value.data.split_manifest import SplitSample, build_split_manifest, select_split
from marginal_value.eval.ablation_eval import summarize_ranked_scores
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.models.learned_linear_ranker import (
    feature_matrix_from_rows,
    load_linear_ranker_model,
    score_linear_ranker,
    sigmoid_scores,
)
from marginal_value.preprocessing.quality import (
    compute_quality_features,
    load_modal_jsonl_imu,
    quality_scores_for_rows,
)
from marginal_value.preprocessing.window_features import compute_window_feature_matrix
from marginal_value.ranking.baseline_ranker import (
    annotate_cluster_features,
    apply_grammar_score_promotion,
    apply_stationary_singleton_guard,
    batch_density,
    build_reason_codes,
    build_scored_rows,
    cluster_cap_rank_rows,
    cluster_aware_rank_rows,
    compute_batch_clusters,
    minmax as _minmax,
    mmr_rank_rows,
    old_knn_novelty,
    parent_prefix_cluster_cap_rank_rows,
    quality_only_rank_rows,
    quality_gated_old_novelty_rank_rows,
    raw_shape_stats_embedding,
    split_large_clusters,
    temporal_order_embedding,
    tiered_cluster_cap_rank_rows,
    window_mean_std_embedding,
)

GRAMMAR_FEATURE_COLUMNS = [
    "token_nll_mean",
    "token_nll_p90",
    "token_nll_p95",
    "transition_nll_mean",
    "transition_nll_p95",
    "rare_bigram_fraction",
    "rare_trigram_fraction",
    "rare_phrase_fraction",
    "longest_unseen_phrase_len",
]

QUALITY_DIAGNOSTIC_COLUMNS = [
    "missing_rate",
    "nan_fraction",
    "inf_fraction",
    "flatline_fraction",
    "saturation_fraction",
    "max_abs_value",
    "extreme_value_fraction",
    "spike_rate",
    "high_frequency_energy",
    "stationary_fraction",
    "axis_imbalance",
    "repeated_timestamp_fraction",
    "timestamp_jitter_fraction",
]


def run_baseline_ranking(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    log_event("baseline_rank", "start", smoke=smoke)
    rng = np.random.default_rng(int(config["ranking"].get("seed", 17)))
    data_root = Path(config["data"]["root"])
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if smoke else "full"
    grammar_features, grammar_feature_path = _load_grammar_features(config, mode=suffix)

    log_event("baseline_rank", "manifest_load_start", root=str(data_root))
    manifest = build_split_manifest(
        data_root,
        pretrain_manifest=config["data"]["pretrain_manifest"],
        val_manifest=config["data"]["val_manifest"],
        extra_manifests=_extra_manifests(config),
        feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
        raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
    )
    support_rows = select_split(manifest, config["splits"]["support_split"])
    query_rows = select_split(manifest, config["splits"]["query_split"])
    negative_rows = select_split(manifest, config["splits"].get("negative_split", "pretrain"))
    min_query_samples = config["ranking"].get("min_query_samples")
    if min_query_samples is not None and len(query_rows) < int(min_query_samples):
        raise ValueError(
            f"query_split '{config['splits']['query_split']}' has {len(query_rows)} cached samples; "
            f"expected at least {int(min_query_samples)}."
        )

    if smoke:
        support_rows = support_rows[: int(config["ranking"].get("smoke_support_samples", 256))]
        query_rows = query_rows[: int(config["ranking"].get("smoke_query_samples", 64))]
        negative_rows = negative_rows[-int(config["ranking"].get("smoke_negative_samples", 64)) :]
    else:
        negative_size = min(int(config["ranking"].get("negative_sample_size", len(query_rows))), len(negative_rows))
        negative_indices = rng.choice(len(negative_rows), size=negative_size, replace=False)
        negative_rows = [negative_rows[int(idx)] for idx in negative_indices]
    log_event(
        "baseline_rank",
        "manifest_ready",
        n_manifest=len(manifest),
        n_support=len(support_rows),
        n_query=len(query_rows),
        n_negative=len(negative_rows),
    )

    embedding_lookup = _build_embedding_lookup(config)
    support_embeddings = _load_embeddings(config, support_rows, label="support", embedding_lookup=embedding_lookup)
    query_embeddings = _load_embeddings(config, query_rows, label="query", embedding_lookup=embedding_lookup)
    negative_embeddings = _load_embeddings(config, negative_rows, label="negative", embedding_lookup=embedding_lookup)

    k_old = int(config["ranking"]["k_old"])
    k_new_density = int(config["ranking"]["k_new_density"])
    novelty_weight = float(config["ranking"].get("novelty_weight", 0.75))
    mmr_lambda = float(config["ranking"].get("mmr_lambda", 0.25))
    reranker_method = str(config["ranking"].get("reranker_method", "cluster_aware"))
    cluster_similarity_threshold = float(config["ranking"].get("cluster_similarity_threshold", 0.985))
    cluster_bonus_weight = float(config["ranking"].get("cluster_bonus_weight", 0.25))
    cluster_cap_top_k = int(config["ranking"].get("cluster_cap_top_k", 200))
    cluster_max_per_cluster = int(config["ranking"].get("cluster_max_per_cluster", 3))
    cluster_cap_key = str(config["ranking"].get("cluster_cap_key", "new_cluster_id"))
    cluster_cap_min_quality = float(config["ranking"].get("cluster_cap_min_quality", 0.0))
    cluster_cap_schedule = config["ranking"].get("cluster_cap_schedule")
    prefix_cluster_cap_top_k = int(config["ranking"].get("prefix_cluster_cap_top_k", 75))
    prefix_cluster_cap_key = str(config["ranking"].get("prefix_cluster_cap_key", "new_cluster_parent_id"))
    prefix_cluster_max_per_cluster = int(config["ranking"].get("prefix_cluster_max_per_cluster", cluster_max_per_cluster))
    quality_gate_threshold = float(config["ranking"].get("quality_gate_threshold", 0.45))
    max_stationary_fraction = config["ranking"].get("max_stationary_fraction")
    max_stationary_fraction = float(max_stationary_fraction) if max_stationary_fraction is not None else None
    max_abs_value = config["ranking"].get("max_abs_value")
    max_abs_value = float(max_abs_value) if max_abs_value is not None else None
    source_cap = config["ranking"].get("source_cap")
    source_cap = int(source_cap) if source_cap is not None else None
    source_cap_key = str(config["ranking"].get("source_cap_key", "source_group_id"))
    sample_rate = float(config.get("quality", {}).get("sample_rate", 30.0))
    max_quality_samples = config.get("quality", {}).get("max_samples_per_clip")
    if max_quality_samples is not None:
        max_quality_samples = int(max_quality_samples)

    query_old_distance, _query_neighbors = old_knn_novelty(support_embeddings, query_embeddings, k=k_old)
    log_event("baseline_rank", "query_old_knn_done", n_query=len(query_rows), k_old=k_old)
    query_density = batch_density(query_embeddings, k=k_new_density)
    log_event("baseline_rank", "query_density_done", n_query=len(query_rows), k_new_density=k_new_density)
    query_quality, query_quality_metadata = quality_scores_for_rows(
        query_rows,
        sample_rate=sample_rate,
        max_samples=max_quality_samples,
        log_component="baseline_rank",
        log_label="query",
    )
    query_scored = build_scored_rows(
        sample_ids=[row.sample_id for row in query_rows],
        embeddings=query_embeddings,
        old_knn_distance=query_old_distance,
        new_density=query_density,
        quality_scores=query_quality,
        novelty_weight=novelty_weight,
    )
    query_scored = _join_quality_metadata(query_scored, query_quality_metadata)
    _join_source_group_ids(query_scored, query_rows)
    query_scored = _join_grammar_features(query_scored, grammar_features)
    query_cluster_ids = compute_batch_clusters(query_embeddings, similarity_threshold=cluster_similarity_threshold)
    log_event("baseline_rank", "query_clusters_done", **_cluster_summary(query_cluster_ids))
    query_scored = annotate_cluster_features(query_scored, query_embeddings, query_cluster_ids)
    query_scored, query_grammar_promotion = _maybe_promote_grammar_scores(config, query_scored, label="query")
    query_scored, query_learned_ranker = _maybe_apply_learned_ranker_scores(config, query_scored, label="query")
    query_scored, query_score_guards = _maybe_apply_score_guards(config, query_scored, label="query")
    query_scored, query_large_cluster_split = _maybe_split_large_clusters(
        config,
        query_scored,
        query_embeddings,
        label="query",
    )
    query_ranked = _rank_rows(
        query_scored,
        query_embeddings,
        reranker_method=reranker_method,
        mmr_lambda=mmr_lambda,
        cluster_bonus_weight=cluster_bonus_weight,
        cluster_cap_top_k=cluster_cap_top_k,
        cluster_max_per_cluster=cluster_max_per_cluster,
        cluster_cap_key=cluster_cap_key,
        cluster_cap_min_quality=cluster_cap_min_quality,
        cluster_cap_schedule=cluster_cap_schedule,
        prefix_cluster_cap_top_k=prefix_cluster_cap_top_k,
        prefix_cluster_cap_key=prefix_cluster_cap_key,
        prefix_cluster_max_per_cluster=prefix_cluster_max_per_cluster,
        quality_gate_threshold=quality_gate_threshold,
        max_stationary_fraction=max_stationary_fraction,
        max_abs_value=max_abs_value,
        source_cap=source_cap,
        source_cap_key=source_cap_key,
    )
    log_event(
        "baseline_rank",
        "query_rerank_done",
        reranker_method=reranker_method,
        cluster_cap_key=cluster_cap_key,
        cluster_cap_min_quality=cluster_cap_min_quality,
        prefix_cluster_cap_key=prefix_cluster_cap_key,
        n_ranked=len(query_ranked),
    )

    log_event("baseline_rank", "candidate_eval_start")
    candidates = _build_candidate_eval(
        support_embeddings=support_embeddings,
        positive_rows=query_rows,
        positive_embeddings=query_embeddings,
        negative_rows=negative_rows,
        negative_embeddings=negative_embeddings,
        k_old=k_old,
        k_new_density=k_new_density,
        novelty_weight=novelty_weight,
        mmr_lambda=mmr_lambda,
        reranker_method=reranker_method,
        cluster_similarity_threshold=cluster_similarity_threshold,
        cluster_bonus_weight=cluster_bonus_weight,
        cluster_cap_top_k=cluster_cap_top_k,
        cluster_max_per_cluster=cluster_max_per_cluster,
        cluster_cap_key=cluster_cap_key,
        cluster_cap_min_quality=cluster_cap_min_quality,
        cluster_cap_schedule=cluster_cap_schedule,
        prefix_cluster_cap_top_k=prefix_cluster_cap_top_k,
        prefix_cluster_cap_key=prefix_cluster_cap_key,
        prefix_cluster_max_per_cluster=prefix_cluster_max_per_cluster,
        quality_gate_threshold=quality_gate_threshold,
        max_stationary_fraction=max_stationary_fraction,
        max_abs_value=max_abs_value,
        source_cap=source_cap,
        source_cap_key=source_cap_key,
        k_values=[int(k) for k in config["ranking"].get("k_values", [10, 50, 100, 200])],
        sample_rate=sample_rate,
        max_quality_samples=max_quality_samples,
        grammar_features=grammar_features,
        config=config,
    )

    submission_path = output_dir / f"baseline_submission_val_{suffix}.csv"
    diagnostics_path = output_dir / f"baseline_diagnostics_val_{suffix}.csv"
    candidate_path = output_dir / f"baseline_ranking_val_candidates_{suffix}.csv"
    quality_path = output_dir / f"baseline_quality_metadata_{suffix}.csv"
    selection_trace_path = output_dir / f"baseline_selection_trace_val_{suffix}.csv"
    candidate_selection_trace_path = output_dir / f"baseline_candidate_selection_trace_val_{suffix}.csv"
    report_path = output_dir / f"baseline_ranking_report_{suffix}.json"

    _write_submission(submission_path, query_ranked)
    _write_rows(diagnostics_path, query_ranked)
    _write_rows(candidate_path, candidates["ranked_rows"])
    _write_rows(quality_path, candidates["quality_metadata"])
    _write_rows(selection_trace_path, _selection_trace_rows(query_ranked))
    _write_rows(candidate_selection_trace_path, _selection_trace_rows(candidates["ranked_rows"]))
    log_event("baseline_rank", "artifacts_written", output_dir=str(output_dir), suffix=suffix)

    report = {
        "mode": suffix,
        "stage": "phase_a_milestone_4_baseline_ranking",
        "representation": config["ranking"]["representation"],
        "support_split": config["splits"]["support_split"],
        "query_split": config["splits"]["query_split"],
        "n_support": len(support_rows),
        "n_query": len(query_rows),
        "n_negative": len(negative_rows),
        "metrics": candidates["metrics"],
        "quality": _quality_summary(query_quality),
        "new_batch_clusters": _cluster_summary(query_cluster_ids),
        "grammar_features": {
            "enabled": bool(config.get("grammar_features", {}).get("enabled", False)),
            "source_path": str(grammar_feature_path) if grammar_feature_path is not None else None,
            "source_paths": sorted(
                {
                    str(row.get("grammar_feature_source_path", ""))
                    for row in grammar_features.values()
                    if row.get("grammar_feature_source_path", "")
                }
            ),
            "n_rows": len(grammar_features),
            "matched_query_rows": int(sum(bool(row.get("grammar_feature_present", False)) for row in query_ranked)),
            "diagnostics_only": not bool(config.get("grammar_features", {}).get("use_in_score", False)),
            "use_in_score": bool(config.get("grammar_features", {}).get("use_in_score", False)),
            "score_variant": config.get("grammar_features", {}).get("score_variant"),
            "score_weight": config.get("grammar_features", {}).get("score_weight"),
            "query_promotion": query_grammar_promotion,
            "candidate_eval_promotion": candidates["grammar_promotion"],
        },
        "corruption_eval": candidates["corruption_eval"],
        "learned_ranker": {
            "query": query_learned_ranker,
            "candidate_eval": candidates["learned_ranker"],
        },
        "score_guards": {
            "query": query_score_guards,
            "candidate_eval": candidates["score_guards"],
        },
        "large_cluster_split": {
            "query": query_large_cluster_split,
            "candidate_eval": candidates["large_cluster_split"],
        },
        "artifacts": {
            "submission": str(submission_path),
            "diagnostics": str(diagnostics_path),
            "candidate_scores": str(candidate_path),
            "quality_metadata": str(quality_path),
            "selection_trace": str(selection_trace_path),
            "candidate_selection_trace": str(candidate_selection_trace_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": suffix,
        "stage": report["stage"],
        "n_support": report["n_support"],
        "n_query": report["n_query"],
        "n_negative": report["n_negative"],
        "ndcg@100": report["metrics"].get("ndcg@100", report["metrics"].get("ndcg@50", 0.0)),
        "precision@100": report["metrics"].get("precision@100", report["metrics"].get("precision@50", 0.0)),
        "report_path": str(report_path),
        "submission_path": str(submission_path),
        "diagnostics_path": str(diagnostics_path),
    }
    log_event("baseline_rank", "done", **result)
    return result


def _build_candidate_eval(
    *,
    support_embeddings: np.ndarray,
    positive_rows: list[SplitSample],
    positive_embeddings: np.ndarray,
    negative_rows: list[SplitSample],
    negative_embeddings: np.ndarray,
    k_old: int,
    k_new_density: int,
    novelty_weight: float,
    mmr_lambda: float,
    reranker_method: str,
    cluster_similarity_threshold: float,
    cluster_bonus_weight: float,
    cluster_cap_top_k: int,
    cluster_max_per_cluster: int,
    cluster_cap_key: str,
    cluster_cap_min_quality: float,
    prefix_cluster_cap_top_k: int,
    prefix_cluster_cap_key: str,
    prefix_cluster_max_per_cluster: int,
    k_values: list[int],
    sample_rate: float,
    max_quality_samples: int | None,
    grammar_features: dict[str, dict[str, object]] | None = None,
    config: dict[str, Any] | None = None,
    cluster_cap_schedule: list[dict[str, object]] | None = None,
    quality_gate_threshold: float = 0.45,
    max_stationary_fraction: float | None = None,
    max_abs_value: float | None = None,
    source_cap: int | None = None,
    source_cap_key: str = "source_group_id",
) -> dict[str, Any]:
    rng = np.random.default_rng(int((config or {}).get("ranking", {}).get("seed", 17)))
    log_event(
        "baseline_rank",
        "candidate_eval_rows_ready",
        n_positive=len(positive_rows),
        n_negative=len(negative_rows),
    )
    clean_sample_rows = positive_rows + negative_rows
    clean_embeddings = np.vstack([positive_embeddings, negative_embeddings])
    clean_labels = np.concatenate([np.ones(len(positive_rows), dtype=int), np.zeros(len(negative_rows), dtype=int)])
    corruption_eval = _build_corruption_eval(
        positive_rows=positive_rows,
        positive_embeddings=positive_embeddings,
        config=config or {},
        rng=rng,
        sample_rate=sample_rate,
        max_samples=max_quality_samples,
    )
    embeddings = np.vstack([clean_embeddings, corruption_eval["embeddings"]]) if len(corruption_eval["embeddings"]) else clean_embeddings
    labels = np.concatenate([clean_labels, corruption_eval["labels"]]) if len(corruption_eval["labels"]) else clean_labels
    is_corruption = np.concatenate(
        [
            np.zeros(len(clean_labels), dtype=bool),
            corruption_eval["is_corruption"],
        ]
    )
    sample_ids = [row.sample_id for row in clean_sample_rows] + list(corruption_eval["sample_ids"])
    sample_splits = [row.split for row in clean_sample_rows] + list(corruption_eval["splits"])
    old_distance, _neighbors = old_knn_novelty(support_embeddings, embeddings, k=k_old)
    log_event("baseline_rank", "candidate_old_knn_done", n_candidates=len(sample_ids), k_old=k_old)
    density = batch_density(embeddings, k=k_new_density)
    log_event("baseline_rank", "candidate_density_done", n_candidates=len(sample_ids), k_new_density=k_new_density)
    positive_quality, positive_quality_metadata = quality_scores_for_rows(
        positive_rows,
        sample_rate=sample_rate,
        max_samples=max_quality_samples,
        log_component="baseline_rank",
        log_label="candidate_positive",
    )
    negative_quality, negative_quality_metadata = quality_scores_for_rows(
        negative_rows,
        sample_rate=sample_rate,
        max_samples=max_quality_samples,
        log_component="baseline_rank",
        log_label="candidate_negative",
    )
    quality_scores = np.concatenate([positive_quality, negative_quality, corruption_eval["quality_scores"]])
    scored_rows = build_scored_rows(
        sample_ids=sample_ids,
        embeddings=embeddings,
        old_knn_distance=old_distance,
        new_density=density,
        quality_scores=quality_scores,
        novelty_weight=novelty_weight,
    )
    combined_quality_metadata = positive_quality_metadata + negative_quality_metadata + corruption_eval["quality_metadata"]
    scored_rows = _join_quality_metadata(scored_rows, combined_quality_metadata)
    clean_source_group_ids = [_source_group_id(row.url) for row in clean_sample_rows]
    corruption_source_group_ids = [str(source_id) for source_id in corruption_eval["source_sample_ids"]]
    for idx, row in enumerate(scored_rows):
        row["candidate_index"] = idx
        row["split"] = sample_splits[idx]
        row["label"] = int(labels[idx])
        row["is_corruption"] = bool(is_corruption[idx])
        row["source_group_id"] = (
            clean_source_group_ids[idx]
            if idx < len(clean_source_group_ids)
            else corruption_source_group_ids[idx - len(clean_source_group_ids)]
        )
        if bool(is_corruption[idx]):
            row["corruption_mode"] = corruption_eval["modes"][idx - len(clean_labels)]
            row["source_sample_id"] = corruption_eval["source_sample_ids"][idx - len(clean_labels)]
    scored_rows = _join_grammar_features(scored_rows, grammar_features or {})
    cluster_ids = compute_batch_clusters(embeddings, similarity_threshold=cluster_similarity_threshold)
    log_event("baseline_rank", "candidate_clusters_done", **_cluster_summary(cluster_ids))
    scored_rows = annotate_cluster_features(scored_rows, embeddings, cluster_ids)
    grammar_promotion = _candidate_grammar_promotion_summary(scored_rows, config or {})
    if grammar_promotion["enabled"] and grammar_promotion["applied"]:
        scored_rows = apply_grammar_score_promotion(scored_rows, **_grammar_promotion_kwargs(config or {}))
        log_event("baseline_rank", "candidate_grammar_promotion_done", **grammar_promotion)
    elif grammar_promotion["enabled"]:
        log_event("baseline_rank", "candidate_grammar_promotion_skipped", **grammar_promotion)
    scored_rows, learned_ranker = _maybe_apply_learned_ranker_scores(config or {}, scored_rows, label="candidate_eval")
    scored_rows, score_guards = _maybe_apply_score_guards(config or {}, scored_rows, label="candidate_eval")
    scored_rows, large_cluster_split = _maybe_split_large_clusters(
        config or {},
        scored_rows,
        embeddings,
        label="candidate_eval",
    )

    ranked_rows = _rank_rows(
        scored_rows,
        embeddings,
        reranker_method=reranker_method,
        mmr_lambda=mmr_lambda,
        cluster_bonus_weight=cluster_bonus_weight,
        cluster_cap_top_k=cluster_cap_top_k,
        cluster_max_per_cluster=cluster_max_per_cluster,
        cluster_cap_key=cluster_cap_key,
        cluster_cap_min_quality=cluster_cap_min_quality,
        cluster_cap_schedule=cluster_cap_schedule,
        prefix_cluster_cap_top_k=prefix_cluster_cap_top_k,
        prefix_cluster_cap_key=prefix_cluster_cap_key,
        prefix_cluster_max_per_cluster=prefix_cluster_max_per_cluster,
        quality_gate_threshold=quality_gate_threshold,
        max_stationary_fraction=max_stationary_fraction,
        max_abs_value=max_abs_value,
        source_cap=source_cap,
        source_cap_key=source_cap_key,
    )
    log_event(
        "baseline_rank",
        "candidate_rerank_done",
        reranker_method=reranker_method,
        cluster_cap_key=cluster_cap_key,
        cluster_cap_min_quality=cluster_cap_min_quality,
        prefix_cluster_cap_key=prefix_cluster_cap_key,
        n_ranked=len(ranked_rows),
    )
    order = np.asarray([int(row["candidate_index"]) for row in ranked_rows], dtype=int)
    metrics = summarize_ranked_scores(
        scores=np.arange(len(order), 0, -1, dtype=float),
        labels=labels[order],
        clusters=cluster_ids[order],
        corruptions=is_corruption[order],
        k_values=k_values,
    )
    return {
        "ranked_rows": ranked_rows,
        "metrics": metrics,
        "quality_metadata": combined_quality_metadata,
        "grammar_promotion": grammar_promotion,
        "learned_ranker": learned_ranker,
        "score_guards": score_guards,
        "large_cluster_split": large_cluster_split,
        "corruption_eval": corruption_eval["summary"],
    }


def _load_window_mean_std_embeddings(rows: Iterable[SplitSample], *, label: str, workers: int = 1) -> np.ndarray:
    rows = list(rows)
    total = len(rows)
    if not rows:
        raise ValueError("No feature rows were available for baseline ranking.")
    workers = max(1, int(workers))
    log_event("baseline_rank", "embedding_load_start", label=label, total=total, workers=workers)
    progress_every = max(1, total // 20) if total else 1
    embeddings: list[np.ndarray | None] = [None] * total
    if workers == 1 or total <= 1:
        for index, row in enumerate(rows, start=1):
            embeddings[index - 1] = _load_one_window_mean_std_embedding(row)
            log_progress("baseline_rank", "embedding_load_progress", index=index, total=total, every=progress_every, label=label)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = {executor.submit(_load_one_window_mean_std_embedding, row): idx for idx, row in enumerate(rows)}
            for completed, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as exc:
                    row = rows[idx]
                    raise RuntimeError(
                        f"Failed loading feature embedding for {label} row {idx + 1}/{total} "
                        f"sample_id={row.sample_id} path={row.feature_path}"
                    ) from exc
                log_progress(
                    "baseline_rank",
                    "embedding_load_progress",
                    index=completed,
                    total=total,
                    every=progress_every,
                    label=label,
                )
    loaded = [embedding for embedding in embeddings if embedding is not None]
    if len(loaded) != total:
        raise RuntimeError(f"Loaded {len(loaded)} embeddings for {total} {label} rows.")
    log_event("baseline_rank", "embedding_load_done", label=label, total=total)
    return np.vstack(loaded)


def _load_one_window_mean_std_embedding(row: SplitSample) -> np.ndarray:
    with np.load(row.feature_path) as data:
        return window_mean_std_embedding(np.asarray(data["window_features"], dtype="float32"))


def _load_temporal_order_embeddings(rows: Iterable[SplitSample], *, label: str, workers: int = 1) -> np.ndarray:
    return _load_derived_embeddings(
        rows,
        label=label,
        workers=workers,
        event_label="temporal_order_embedding",
        load_one=_load_one_temporal_order_embedding,
    )


def _load_one_temporal_order_embedding(row: SplitSample) -> np.ndarray:
    with np.load(row.feature_path) as data:
        return temporal_order_embedding(np.asarray(data["window_features"], dtype="float32"))


def _load_raw_shape_stats_embeddings(
    rows: Iterable[SplitSample],
    *,
    label: str,
    workers: int = 1,
    sample_rate: float = 30.0,
    max_samples: int | None = None,
) -> np.ndarray:
    return _load_derived_embeddings(
        rows,
        label=label,
        workers=workers,
        event_label="raw_shape_stats_embedding",
        load_one=lambda row: _load_one_raw_shape_stats_embedding(
            row,
            sample_rate=sample_rate,
            max_samples=max_samples,
        ),
    )


def _load_one_raw_shape_stats_embedding(
    row: SplitSample,
    *,
    sample_rate: float,
    max_samples: int | None,
) -> np.ndarray:
    samples, _timestamps = load_modal_jsonl_imu(row.raw_path, max_samples=max_samples)
    return raw_shape_stats_embedding(samples, sample_rate=sample_rate)


def _load_derived_embeddings(
    rows: Iterable[SplitSample],
    *,
    label: str,
    workers: int,
    event_label: str,
    load_one,
) -> np.ndarray:
    rows = list(rows)
    total = len(rows)
    if not rows:
        raise ValueError("No rows were available for baseline ranking.")
    workers = max(1, int(workers))
    log_event("baseline_rank", f"{event_label}_load_start", label=label, total=total, workers=workers)
    progress_every = max(1, total // 20) if total else 1
    embeddings: list[np.ndarray | None] = [None] * total
    if workers == 1 or total <= 1:
        for index, row in enumerate(rows, start=1):
            embeddings[index - 1] = load_one(row)
            log_progress(
                "baseline_rank",
                f"{event_label}_load_progress",
                index=index,
                total=total,
                every=progress_every,
                label=label,
            )
    else:
        with ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = {executor.submit(load_one, row): idx for idx, row in enumerate(rows)}
            for completed, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as exc:
                    row = rows[idx]
                    raise RuntimeError(
                        f"Failed loading {event_label} for {label} row {idx + 1}/{total} "
                        f"sample_id={row.sample_id}"
                    ) from exc
                log_progress(
                    "baseline_rank",
                    f"{event_label}_load_progress",
                    index=completed,
                    total=total,
                    every=progress_every,
                    label=label,
                )
    loaded = [embedding for embedding in embeddings if embedding is not None]
    if len(loaded) != total:
        raise RuntimeError(f"Loaded {len(loaded)} {event_label} values for {total} {label} rows.")
    log_event("baseline_rank", f"{event_label}_load_done", label=label, total=total)
    return np.vstack(loaded)


def _load_embeddings(
    config: dict[str, Any],
    rows: Iterable[SplitSample],
    *,
    label: str,
    embedding_lookup: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    representation = str(config["ranking"].get("representation", "window_mean_std_pool"))
    if representation == "window_mean_std_pool":
        return _load_window_mean_std_embeddings(
            rows,
            label=label,
            workers=int(config["ranking"].get("embedding_load_workers", 1)),
        )
    if representation == "temporal_order":
        return _load_temporal_order_embeddings(
            rows,
            label=label,
            workers=int(config["ranking"].get("embedding_load_workers", 1)),
        )
    if representation == "raw_shape_stats":
        max_samples = config.get("ranking", {}).get("raw_shape_max_samples")
        if max_samples is None:
            max_samples = config.get("quality", {}).get("max_samples_per_clip")
        return _load_raw_shape_stats_embeddings(
            rows,
            label=label,
            workers=int(config["ranking"].get("embedding_load_workers", 1)),
            sample_rate=float(config.get("quality", {}).get("sample_rate", 30.0)),
            max_samples=int(max_samples) if max_samples is not None else None,
        )
    if representation == "encoder_artifact":
        if embedding_lookup is None:
            raise ValueError("encoder_artifact representation requires an embedding lookup.")
        rows = list(rows)
        missing = [row.sample_id for row in rows if row.sample_id not in embedding_lookup]
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Missing encoder artifact embeddings for {len(missing)} {label} rows: {preview}")
        log_event("baseline_rank", "embedding_artifact_load_done", label=label, total=len(rows))
        return np.vstack([embedding_lookup[row.sample_id] for row in rows]).astype(float)
    raise ValueError(f"Unsupported ranking representation: {representation}")


def _build_embedding_lookup(config: dict[str, Any]) -> dict[str, np.ndarray] | None:
    if str(config["ranking"].get("representation", "window_mean_std_pool")) != "encoder_artifact":
        return None
    embedding_config = config.get("encoder_embeddings", {})
    if not isinstance(embedding_config, dict):
        raise ValueError("encoder_artifact representation requires encoder_embeddings config.")
    lookup: dict[str, np.ndarray] = {}
    for prefix in ("support", "query"):
        embeddings_path = Path(str(embedding_config[f"{prefix}_embeddings"]))
        manifest_path = Path(str(embedding_config[f"{prefix}_manifest"]))
        log_event(
            "baseline_rank",
            "embedding_artifact_read_start",
            prefix=prefix,
            embeddings_path=str(embeddings_path),
            manifest_path=str(manifest_path),
        )
        prefix_lookup = _read_embedding_artifact(embeddings_path, manifest_path)
        overlap = set(lookup) & set(prefix_lookup)
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Duplicate sample IDs across encoder artifacts: {preview}")
        lookup.update(prefix_lookup)
        log_event("baseline_rank", "embedding_artifact_read_done", prefix=prefix, n_rows=len(prefix_lookup))
    return lookup


def _read_embedding_artifact(embeddings_path: Path, manifest_path: Path) -> dict[str, np.ndarray]:
    embeddings = np.asarray(np.load(embeddings_path), dtype=float)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if embeddings.ndim != 2:
        raise ValueError(f"Encoder embeddings must be a 2D array: {embeddings_path}")
    if len(rows) != len(embeddings):
        raise ValueError(
            f"Encoder manifest row count does not match embeddings: {manifest_path} has {len(rows)}, "
            f"{embeddings_path} has {len(embeddings)}"
        )
    lookup: dict[str, np.ndarray] = {}
    for row, embedding in zip(rows, embeddings):
        sample_id = str(row.get("sample_id") or row.get("worker_id") or "")
        if not sample_id:
            continue
        lookup[sample_id] = embedding
    if len(lookup) != len(embeddings):
        raise ValueError(f"Encoder manifest contains missing or duplicate sample IDs: {manifest_path}")
    return lookup


def _build_corruption_eval(
    *,
    positive_rows: list[SplitSample],
    positive_embeddings: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
    sample_rate: float = 30.0,
    max_samples: int | None = None,
) -> dict[str, Any]:
    corruption_config = config.get("corruption_eval", {})
    if not isinstance(corruption_config, dict) or not bool(corruption_config.get("enabled", False)):
        return _empty_corruption_eval("disabled")
    if len(positive_rows) == 0:
        return _empty_corruption_eval("no_positive_rows")

    sample_size = min(int(corruption_config.get("sample_size", 128)), len(positive_rows))
    if sample_size <= 0:
        return _empty_corruption_eval("sample_size_zero")
    quality_score = float(corruption_config.get("quality_score", 0.05))
    quality_score = float(np.clip(quality_score, 0.0, 1.0))
    modes = [str(mode) for mode in corruption_config.get("modes", ["flatline", "spike", "saturation", "jitter"])]
    if not modes:
        modes = ["flatline"]
    indices = rng.choice(len(positive_rows), size=sample_size, replace=False)
    if bool(corruption_config.get("raw_signal", False)):
        representation = str(config.get("ranking", {}).get("representation", "window_mean_std_pool"))
        if representation != "window_mean_std_pool":
            raise ValueError("corruption_eval.raw_signal requires window_mean_std_pool representation.")
        return _build_raw_signal_corruption_eval(
            positive_rows=positive_rows,
            selected_indices=indices,
            modes=modes,
            rng=rng,
            sample_rate=sample_rate,
            max_samples=max_samples,
        )

    base = np.asarray(positive_embeddings, dtype=float)
    scale = np.maximum(np.std(base, axis=0), 1.0e-6)

    embeddings: list[np.ndarray] = []
    sample_ids: list[str] = []
    source_sample_ids: list[str] = []
    selected_modes: list[str] = []
    quality_metadata: list[dict[str, float | str]] = []
    for out_idx, source_idx in enumerate(indices):
        source_row = positive_rows[int(source_idx)]
        mode = modes[out_idx % len(modes)]
        corrupted = _corrupt_embedding(base[int(source_idx)], mode=mode, scale=scale, rng=rng)
        sample_id = f"{source_row.sample_id}__corrupt_{mode}_{out_idx:04d}"
        embeddings.append(corrupted)
        sample_ids.append(sample_id)
        source_sample_ids.append(source_row.sample_id)
        selected_modes.append(mode)
        quality_metadata.append(
            {
                "sample_id": sample_id,
                "split": "corruption",
                "raw_path": str(source_row.raw_path),
                "source_sample_id": source_row.sample_id,
                "corruption_mode": mode,
                "quality_score": quality_score,
                "missing_rate": 0.0,
                "flatline_fraction": 1.0 if mode == "flatline" else 0.0,
                "saturation_fraction": 1.0 if mode == "saturation" else 0.0,
                "spike_rate": 1.0 if mode in {"spike", "jitter"} else 0.0,
            }
        )

    log_event(
        "baseline_rank",
        "corruption_eval_ready",
        enabled=True,
        n_corruptions=len(embeddings),
        modes=sorted(set(selected_modes)),
        quality_score=quality_score,
    )
    return {
        "embeddings": np.vstack(embeddings).astype(float),
        "labels": np.zeros(len(embeddings), dtype=int),
        "is_corruption": np.ones(len(embeddings), dtype=bool),
        "quality_scores": np.full(len(embeddings), quality_score, dtype=float),
        "sample_ids": sample_ids,
        "source_sample_ids": source_sample_ids,
        "splits": ["corruption"] * len(embeddings),
        "modes": selected_modes,
        "quality_metadata": quality_metadata,
        "summary": {
            "enabled": True,
            "n_corruptions": len(embeddings),
            "quality_score": quality_score,
            "modes": sorted(set(selected_modes)),
            "raw_signal": False,
            "quality_score_source": "configured_constant",
        },
    }


def _build_raw_signal_corruption_eval(
    *,
    positive_rows: list[SplitSample],
    selected_indices: np.ndarray,
    modes: list[str],
    rng: np.random.Generator,
    sample_rate: float,
    max_samples: int | None,
) -> dict[str, Any]:
    embeddings: list[np.ndarray] = []
    quality_scores: list[float] = []
    sample_ids: list[str] = []
    source_sample_ids: list[str] = []
    selected_modes: list[str] = []
    quality_metadata: list[dict[str, float | str]] = []
    for out_idx, source_idx in enumerate(selected_indices):
        source_row = positive_rows[int(source_idx)]
        mode = modes[out_idx % len(modes)]
        samples, timestamps = load_modal_jsonl_imu(source_row.raw_path, max_samples=max_samples)
        corrupted_samples, corrupted_timestamps = _corrupt_raw_signal(
            samples,
            timestamps=timestamps,
            mode=mode,
            sample_rate=sample_rate,
            rng=rng,
        )
        window_features = compute_window_feature_matrix(corrupted_samples)
        embedding = window_mean_std_embedding(window_features)
        features = compute_quality_features(corrupted_samples, timestamps=corrupted_timestamps, sample_rate=sample_rate)
        quality_score = float(features["quality_score"])
        sample_id = f"{source_row.sample_id}__corrupt_{mode}_{out_idx:04d}"

        embeddings.append(embedding)
        quality_scores.append(quality_score)
        sample_ids.append(sample_id)
        source_sample_ids.append(source_row.sample_id)
        selected_modes.append(mode)
        metadata_row: dict[str, float | str] = {
            "sample_id": sample_id,
            "split": "corruption",
            "raw_path": str(source_row.raw_path),
            "source_sample_id": source_row.sample_id,
            "corruption_mode": mode,
            "quality_score_source": "computed_from_corrupted_raw",
        }
        metadata_row.update({key: float(value) for key, value in features.items()})
        quality_metadata.append(metadata_row)

    scores = np.asarray(quality_scores, dtype=float)
    log_event(
        "baseline_rank",
        "corruption_eval_ready",
        enabled=True,
        raw_signal=True,
        n_corruptions=len(embeddings),
        modes=sorted(set(selected_modes)),
        quality_mean=float(np.mean(scores)) if len(scores) else 0.0,
        quality_min=float(np.min(scores)) if len(scores) else 0.0,
    )
    return {
        "embeddings": np.vstack(embeddings).astype(float),
        "labels": np.zeros(len(embeddings), dtype=int),
        "is_corruption": np.ones(len(embeddings), dtype=bool),
        "quality_scores": scores,
        "sample_ids": sample_ids,
        "source_sample_ids": source_sample_ids,
        "splits": ["corruption"] * len(embeddings),
        "modes": selected_modes,
        "quality_metadata": quality_metadata,
        "summary": {
            "enabled": True,
            "n_corruptions": len(embeddings),
            "modes": sorted(set(selected_modes)),
            "raw_signal": True,
            "quality_score_source": "computed_from_corrupted_raw",
            "quality_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "quality_min": float(np.min(scores)) if len(scores) else 0.0,
            "quality_max": float(np.max(scores)) if len(scores) else 0.0,
        },
    }


def _corrupt_raw_signal(
    samples: np.ndarray,
    *,
    timestamps: np.ndarray | None,
    mode: str,
    sample_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(samples, dtype=float).copy()
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError("raw corruption requires samples shaped [time, >=6].")
    n_samples, n_channels = values.shape
    n_core_channels = min(6, n_channels)
    output_timestamps = None if timestamps is None else np.asarray(timestamps, dtype=float).copy()

    if mode == "flatline":
        replacement = np.nanmedian(values[:, :n_core_channels], axis=0)
        values[:, :n_core_channels] = replacement
    elif mode == "spike":
        n_events = max(1, n_samples // 45)
        event_indices = rng.choice(n_samples, size=n_events, replace=False)
        channel_indices = rng.integers(0, n_core_channels, size=n_events)
        signs = rng.choice([-1.0, 1.0], size=n_events)
        values[event_indices, channel_indices] += signs * 500.0
    elif mode == "saturation":
        channel = int(rng.integers(0, n_core_channels))
        sign = float(rng.choice([-1.0, 1.0]))
        values[:, channel] = sign * 300.0
    elif mode == "jitter":
        noise = rng.normal(0.0, 12.0, size=(n_samples, n_core_channels))
        alternating = ((np.arange(n_samples) % 2) * 2.0 - 1.0)[:, None]
        values[:, :n_core_channels] += noise * alternating
        base = np.arange(n_samples, dtype=float) / sample_rate
        jitter = rng.normal(0.0, 1.25 / sample_rate, size=n_samples)
        output_timestamps = base + jitter
        output_timestamps = np.maximum.accumulate(output_timestamps)
    else:
        values[:, :n_core_channels] += rng.normal(0.0, 20.0, size=(n_samples, n_core_channels))
    return values, output_timestamps


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


def _corrupt_embedding(
    embedding: np.ndarray,
    *,
    mode: str,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(embedding, dtype=float).copy()
    if mode == "flatline":
        return np.zeros_like(values)
    if mode == "spike":
        corrupted = values.copy()
        n_spikes = max(1, len(values) // 12)
        indices = rng.choice(len(values), size=n_spikes, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_spikes)
        corrupted[indices] += signs * 25.0 * scale[indices]
        return corrupted
    if mode == "saturation":
        limit = np.percentile(np.abs(values), 90) if len(values) else 1.0
        return np.clip(values * 8.0, -max(limit, 1.0), max(limit, 1.0))
    if mode == "jitter":
        return values + rng.normal(0.0, 8.0, size=values.shape) * scale
    return values + rng.normal(0.0, 5.0, size=values.shape) * scale


def _maybe_promote_grammar_scores(
    config: dict[str, Any],
    rows: list[dict[str, object]],
    *,
    label: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not _grammar_promotion_enabled(config):
        return rows, {"enabled": False, "applied": False, "skip_reason": "disabled"}
    promoted = apply_grammar_score_promotion(rows, **_grammar_promotion_kwargs(config))
    applied_count = int(sum(bool(row.get("grammar_promotion_applied", False)) for row in promoted))
    summary = {
        "enabled": True,
        "applied": applied_count > 0,
        "applied_count": applied_count,
        "n_rows": len(promoted),
        "label": label,
        "skip_reason": "" if applied_count > 0 else "no_rows_with_grammar_features",
    }
    log_event("baseline_rank", "grammar_promotion_done", **summary)
    return promoted, summary


def _maybe_apply_learned_ranker_scores(
    config: dict[str, Any],
    rows: list[dict[str, object]],
    *,
    label: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    learned_config = config.get("learned_ranker", {})
    if not isinstance(learned_config, dict) or not bool(learned_config.get("enabled", False)):
        return rows, {"enabled": False, "applied": False, "skip_reason": "disabled", "label": label}
    if not rows:
        return rows, {"enabled": True, "applied": False, "skip_reason": "no_rows", "label": label}

    model_path = Path(str(learned_config.get("model_path", "")))
    if not model_path.exists():
        raise FileNotFoundError(f"Missing learned-ranker model: {model_path}")
    model, feature_names, metadata = load_linear_ranker_model(model_path)
    feature_matrix = feature_matrix_from_rows(rows, feature_names)
    raw_scores = score_linear_ranker(model, feature_matrix)
    score_transform = str(learned_config.get("score_transform", "sigmoid"))
    if score_transform == "sigmoid":
        learned_scores = sigmoid_scores(raw_scores)
    elif score_transform == "minmax":
        learned_scores = _minmax(raw_scores)
    else:
        raise ValueError("learned_ranker.score_transform must be 'sigmoid' or 'minmax'.")
    score_weight = float(learned_config.get("score_weight", 1.0))
    if not 0.0 <= score_weight <= 1.0:
        raise ValueError("learned_ranker.score_weight must be in [0, 1].")

    scored: list[dict[str, object]] = []
    for row, raw_score, learned_score in zip(rows, raw_scores, learned_scores):
        output = dict(row)
        quality = _safe_float(output.get("quality_score", 1.0))
        previous_ranker_score = _safe_float(output.get("ranker_score", output.get("final_score", 0.0)))
        previous_final_score = _safe_float(output.get("final_score", quality * previous_ranker_score))
        ranker_score = (1.0 - score_weight) * previous_ranker_score + score_weight * float(learned_score)
        output["pre_learned_ranker_score"] = previous_ranker_score
        output["pre_learned_final_score"] = previous_final_score
        output["learned_ranker_raw_score"] = float(raw_score)
        output["learned_ranker_score"] = float(learned_score)
        output["learned_ranker_score_weight"] = score_weight
        output["learned_ranker_model_path"] = str(model_path)
        output["learned_ranker_applied"] = True
        output["ranker_score"] = float(np.clip(ranker_score, 0.0, 1.0))
        output["final_score"] = float(np.clip(quality * output["ranker_score"], 0.0, 1.0))
        scored.append(output)

    reasons = build_reason_codes(scored)
    for row, reason in zip(scored, reasons):
        row["reason_code"] = reason

    summary = {
        "enabled": True,
        "applied": True,
        "label": label,
        "model_path": str(model_path),
        "model_feature_count": len(feature_names),
        "score_weight": score_weight,
        "score_transform": score_transform,
        "score_mean": float(np.mean(learned_scores)),
        "score_p90": float(np.percentile(learned_scores, 90)),
        "metadata": metadata,
    }
    log_event("baseline_rank", "learned_ranker_applied", **{key: value for key, value in summary.items() if key != "metadata"})
    return scored, summary


def _maybe_apply_score_guards(
    config: dict[str, Any],
    rows: list[dict[str, object]],
    *,
    label: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    guard_config = config.get("score_guards", {})
    stationary_config = guard_config.get("stationary_singleton", {}) if isinstance(guard_config, dict) else {}
    if not isinstance(stationary_config, dict) or not bool(stationary_config.get("enabled", False)):
        return rows, {"enabled": False, "applied": False, "skip_reason": "disabled", "label": label}
    guarded, summary = apply_stationary_singleton_guard(
        rows,
        stationary_threshold=float(stationary_config.get("stationary_threshold", 0.90)),
        max_new_density_score=float(stationary_config.get("max_new_density_score", 0.35)),
        min_grammar_score=float(stationary_config.get("min_grammar_score", 0.85)),
        penalty_multiplier=float(stationary_config.get("penalty_multiplier", 0.35)),
    )
    summary["label"] = label
    log_event("baseline_rank", "score_guard_stationary_singleton_done", **summary)
    return guarded, summary


def _maybe_split_large_clusters(
    config: dict[str, Any],
    rows: list[dict[str, object]],
    embeddings: np.ndarray | None = None,
    *,
    label: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    split_config = config.get("large_cluster_split", config.get("ranking", {}).get("large_cluster_split", {}))
    if not isinstance(split_config, dict) or not bool(split_config.get("enabled", False)):
        return rows, {"enabled": False, "applied": False, "skip_reason": "disabled", "label": label}
    if not rows:
        return rows, {"enabled": True, "applied": False, "skip_reason": "no_rows", "label": label}
    score_columns = split_config.get("score_columns")
    if score_columns is None:
        score_columns = (
            "grammar_score_component",
            "grammar_score",
            "old_novelty_score",
            "new_density_score",
            "distance_to_new_cluster_medoid",
            "final_score",
        )
    split_rows, summary = split_large_clusters(
        rows,
        embeddings,
        max_cluster_size=int(split_config.get("max_cluster_size", 300)),
        target_subcluster_size=int(split_config.get("target_subcluster_size", 75)),
        score_columns=tuple(str(column) for column in score_columns),
        split_method=str(split_config.get("method", split_config.get("split_method", "feature_kmeans"))),
        score_feature_weight=float(split_config.get("score_feature_weight", 0.0)),
        kmeans_iterations=int(split_config.get("kmeans_iterations", 16)),
    )
    summary["label"] = label
    summary["applied"] = bool(summary.get("n_split_parent_clusters", 0))
    log_event("baseline_rank", "large_cluster_split_done", **summary)
    return split_rows, summary


def _join_quality_metadata(
    rows: Iterable[dict[str, object]],
    quality_metadata: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    metadata_by_id = {str(row.get("sample_id", "")): row for row in quality_metadata if row.get("sample_id", "")}
    joined: list[dict[str, object]] = []
    for row in rows:
        output = dict(row)
        sample_id = str(output.get("sample_id", output.get("worker_id", "")))
        metadata = metadata_by_id.get(sample_id)
        output["quality_metadata_present"] = metadata is not None
        if metadata is not None:
            for column in QUALITY_DIAGNOSTIC_COLUMNS:
                if column in metadata:
                    output[column] = _safe_float(metadata.get(column, 0.0))
            output["quality_metadata_split"] = metadata.get("split", "")
            output["quality_metadata_source"] = metadata.get("quality_score_source", "")
        else:
            for column in QUALITY_DIAGNOSTIC_COLUMNS:
                output.setdefault(column, 0.0)
            output.setdefault("quality_metadata_split", "")
            output.setdefault("quality_metadata_source", "")
        joined.append(output)
    return joined


def _join_source_group_ids(rows: list[dict[str, object]], samples: list[SplitSample]) -> None:
    if len(rows) != len(samples):
        raise ValueError("rows and samples must have the same length for source metadata.")
    for row, sample in zip(rows, samples, strict=True):
        row["source_group_id"] = _source_group_id(sample.url)


def _source_group_id(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
    for part in parts:
        if part.startswith("worker"):
            return part
    return parsed.netloc + ":" + PurePosixPath(parsed.path).name


def _candidate_grammar_promotion_summary(rows: list[dict[str, object]], config: dict[str, Any]) -> dict[str, object]:
    if not _grammar_promotion_enabled(config):
        return {"enabled": False, "applied": False, "skip_reason": "disabled"}
    allowed, reason = _candidate_grammar_promotion_allowed(rows)
    return {
        "enabled": True,
        "applied": allowed,
        "skip_reason": "" if allowed else reason,
    }


def _candidate_grammar_promotion_allowed(rows: Iterable[dict[str, object]]) -> tuple[bool, str]:
    by_label: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        label = int(row.get("label", -1))
        by_label.setdefault(label, []).append(row)
    if set(by_label) != {0, 1}:
        return False, "requires_binary_candidate_labels"
    for label in sorted(by_label):
        if any(not _truthy(row.get("grammar_feature_present", False)) for row in by_label[label]):
            return False, f"missing_grammar_features_for_label_{label}"
    return True, "all_labels_have_grammar_features"


def _grammar_promotion_enabled(config: dict[str, Any]) -> bool:
    grammar_config = config.get("grammar_features", {})
    return isinstance(grammar_config, dict) and bool(grammar_config.get("enabled", False)) and bool(grammar_config.get("use_in_score", False))


def _grammar_promotion_kwargs(config: dict[str, Any]) -> dict[str, object]:
    grammar_config = config.get("grammar_features", {})
    return {
        "score_variant": str(grammar_config.get("score_variant", "grammar_surprisal_mix")),
        "score_weight": float(grammar_config.get("score_weight", 0.0)),
        "min_quality": float(grammar_config.get("min_quality", 0.45)),
        "min_new_density_score": float(grammar_config.get("min_new_density_score", 0.35)),
    }


def _load_grammar_features(config: dict[str, Any], *, mode: str) -> tuple[dict[str, dict[str, object]], Path | None]:
    grammar_config = config.get("grammar_features", {})
    if not isinstance(grammar_config, dict) or not bool(grammar_config.get("enabled", False)):
        return {}, None
    path_templates = _grammar_feature_path_templates(grammar_config)
    feature_paths = []
    seen_paths: set[str] = set()
    for template in path_templates:
        feature_path = Path(template.format(mode=mode))
        path_key = str(feature_path)
        if path_key in seen_paths:
            continue
        feature_paths.append(feature_path)
        seen_paths.add(path_key)
    features: dict[str, dict[str, object]] = {}
    for feature_path in feature_paths:
        log_event("baseline_rank", "grammar_features_load_start", path=str(feature_path), mode=mode)
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing grammar feature CSV: {feature_path}")
        with feature_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_id = row.get("sample_id") or row.get("worker_id")
                if not sample_id:
                    continue
                sample_key = str(sample_id)
                if sample_key in features:
                    previous_split = str(features[sample_key].get("grammar_feature_split", ""))
                    current_split = str(row.get("split", ""))
                    if previous_split == current_split:
                        continue
                    raise ValueError(f"Duplicate grammar features for sample_id '{sample_key}' across configured files.")
                parsed: dict[str, object] = {
                    "grammar_feature_present": True,
                    "grammar_feature_split": row.get("split", ""),
                    "grammar_feature_source_path": str(feature_path),
                }
                for column in GRAMMAR_FEATURE_COLUMNS:
                    parsed[column] = _safe_float(row.get(column, 0.0))
                features[sample_key] = parsed
        log_event("baseline_rank", "grammar_features_file_load_done", path=str(feature_path), n_rows=len(features), mode=mode)
    log_event(
        "baseline_rank",
        "grammar_features_load_done",
        paths=[str(path) for path in feature_paths],
        n_rows=len(features),
        mode=mode,
    )
    return features, feature_paths[0] if feature_paths else None


def _grammar_feature_path_templates(grammar_config: dict[str, Any]) -> list[str]:
    path_template = str(grammar_config.get("path_template", ""))
    templates = [path_template] if path_template else []
    extra_templates = grammar_config.get("extra_path_templates", [])
    if extra_templates:
        if not isinstance(extra_templates, list):
            raise ValueError("grammar_features.extra_path_templates must be a list when provided.")
        templates.extend(str(template) for template in extra_templates)
    if not templates:
        raise ValueError("grammar_features.path_template is required when grammar features are enabled.")
    return templates


def _extra_manifests(config: dict[str, Any]) -> dict[str, str]:
    data_config = config.get("data", {})
    if data_config.get("new_manifest"):
        return {"new": str(data_config["new_manifest"])}
    return {}


def _join_grammar_features(
    rows: Iterable[dict[str, object]],
    grammar_features: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    joined: list[dict[str, object]] = []
    for row in rows:
        output = dict(row)
        lookup_keys = _grammar_lookup_keys(output)
        matched_key = ""
        features = None
        for key in lookup_keys:
            features = grammar_features.get(key)
            if features is not None:
                matched_key = key
                break
        if features is None:
            output["grammar_feature_present"] = False
            output["grammar_feature_split"] = ""
            output["grammar_feature_lookup_key"] = ""
            output["grammar_feature_source_path"] = ""
            for column in GRAMMAR_FEATURE_COLUMNS:
                output[column] = 0.0
        else:
            output.update(features)
            output["grammar_feature_present"] = True
            output.setdefault("grammar_feature_split", "")
            output["grammar_feature_lookup_key"] = matched_key
            for column in GRAMMAR_FEATURE_COLUMNS:
                output.setdefault(column, 0.0)
        joined.append(output)
    return joined


def _grammar_lookup_keys(row: dict[str, object]) -> list[str]:
    raw_values = [
        row.get("sample_id", ""),
        row.get("worker_id", ""),
        row.get("source_sample_id", ""),
    ]
    keys: list[str] = []
    for value in raw_values:
        key = str(value)
        if key and key not in keys:
            keys.append(key)
        if "__corrupt_" in key:
            source_key = key.split("__corrupt_", 1)[0]
            if source_key and source_key not in keys:
                keys.append(source_key)
    return keys


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _rank_rows(
    rows: list[dict[str, object]],
    embeddings: np.ndarray,
    *,
    reranker_method: str,
    mmr_lambda: float,
    cluster_bonus_weight: float,
    cluster_cap_top_k: int = 200,
    cluster_max_per_cluster: int = 3,
    cluster_cap_key: str = "new_cluster_id",
    cluster_cap_min_quality: float = 0.0,
    cluster_cap_schedule: list[dict[str, object]] | None = None,
    prefix_cluster_cap_top_k: int = 75,
    prefix_cluster_cap_key: str = "new_cluster_parent_id",
    prefix_cluster_max_per_cluster: int = 8,
    quality_gate_threshold: float = 0.45,
    max_stationary_fraction: float | None = None,
    max_abs_value: float | None = None,
    source_cap: int | None = None,
    source_cap_key: str = "source_group_id",
) -> list[dict[str, object]]:
    if reranker_method == "quality_only":
        return quality_only_rank_rows(rows)
    if reranker_method == "quality_gated_old_novelty":
        return quality_gated_old_novelty_rank_rows(
            rows,
            quality_threshold=quality_gate_threshold,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
    if reranker_method == "quality_gated_old_novelty_sourcecap":
        return quality_gated_old_novelty_rank_rows(
            rows,
            quality_threshold=quality_gate_threshold,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
            source_cap=source_cap or 1,
            source_key=source_cap_key,
        )
    if reranker_method == "cluster_aware":
        return cluster_aware_rank_rows(
            rows,
            embeddings,
            lambda_redundancy=mmr_lambda,
            cluster_bonus_weight=cluster_bonus_weight,
        )
    if reranker_method == "cluster_cap":
        return cluster_cap_rank_rows(
            rows,
            embeddings,
            lambda_redundancy=mmr_lambda,
            cluster_bonus_weight=cluster_bonus_weight,
            cluster_cap_top_k=cluster_cap_top_k,
            cluster_max_per_cluster=cluster_max_per_cluster,
            cluster_key=cluster_cap_key,
            cluster_cap_min_quality=cluster_cap_min_quality,
        )
    if reranker_method == "parent_prefix_cluster_cap":
        return parent_prefix_cluster_cap_rank_rows(
            rows,
            embeddings,
            lambda_redundancy=mmr_lambda,
            cluster_bonus_weight=cluster_bonus_weight,
            cluster_cap_top_k=cluster_cap_top_k,
            fill_cluster_key=cluster_cap_key,
            fill_max_per_cluster=cluster_max_per_cluster,
            prefix_top_k=prefix_cluster_cap_top_k,
            prefix_cluster_key=prefix_cluster_cap_key,
            prefix_max_per_cluster=prefix_cluster_max_per_cluster,
            cluster_cap_min_quality=cluster_cap_min_quality,
        )
    if reranker_method == "tiered_cluster_cap":
        schedule = cluster_cap_schedule or [
            {"top_k": cluster_cap_top_k, "max_per_cluster": cluster_max_per_cluster}
        ]
        return tiered_cluster_cap_rank_rows(
            rows,
            embeddings,
            cap_schedule=schedule,
            lambda_redundancy=mmr_lambda,
            cluster_bonus_weight=cluster_bonus_weight,
            cluster_key=cluster_cap_key,
            cluster_cap_min_quality=cluster_cap_min_quality,
        )
    return mmr_rank_rows(rows, embeddings, lambda_redundancy=mmr_lambda)


def _cluster_summary(cluster_ids: np.ndarray) -> dict[str, float]:
    ids = np.asarray(cluster_ids, dtype=int)
    if ids.size == 0:
        return {
            "n_clusters": 0.0,
            "singleton_fraction": 0.0,
            "mean_cluster_size": 0.0,
            "max_cluster_size": 0.0,
        }
    _, counts = np.unique(ids, return_counts=True)
    return {
        "n_clusters": float(len(counts)),
        "singleton_fraction": float(np.mean(counts == 1)),
        "mean_cluster_size": float(np.mean(counts)),
        "max_cluster_size": float(np.max(counts)),
    }


def _quality_summary(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return {
            "mean": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "low_quality_fraction": 0.0,
        }
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "low_quality_fraction": float(np.mean(values < 0.45)),
    }


def _write_submission(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = ["worker_id", "rank", "score", "quality_score", "reason_code"]
    _write_csv(path, fieldnames, _submission_rows(rows))


def _selection_trace_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    trace_rows: list[dict[str, object]] = []
    for row in rows:
        trace_rows.append(
            {
                "rank": int(_safe_float(row.get("rank", 0))),
                "worker_id": row.get("worker_id", row.get("sample_id", "")),
                "sample_id": row.get("sample_id", row.get("worker_id", "")),
                "base_score": _safe_float(row.get("phase_a_final_score", row.get("final_score", 0.0))),
                "final_score": _safe_float(row.get("final_score", 0.0)),
                "rerank_score": _safe_float(row.get("rerank_score", row.get("cluster_selection_score", 0.0))),
                "cluster_id": row.get("new_cluster_id", ""),
                "cluster_cap_key": row.get("cluster_cap_key", ""),
                "cluster_cap_cluster_id": row.get("cluster_cap_cluster_id", ""),
                "cluster_cap_min_quality": row.get("cluster_cap_min_quality", ""),
                "new_cluster_parent_id": row.get("new_cluster_parent_id", ""),
                "new_cluster_size": row.get("new_cluster_size", ""),
                "cluster_count_before_selection": int(_safe_float(row.get("cluster_count_before_selection", row.get("cluster_round", 0)))),
                "was_cluster_cap_active": bool(row.get("was_cluster_cap_active", False)),
                "was_selected_by_fallback": bool(row.get("was_selected_by_fallback", False)),
                "eligible_under_cluster_cap_count": int(_safe_float(row.get("eligible_under_cluster_cap_count", 0))),
                "cluster_cap_top_k": row.get("cluster_cap_top_k", ""),
                "cluster_max_per_cluster": row.get("cluster_max_per_cluster", ""),
                "cluster_cap_schedule": row.get("cluster_cap_schedule", ""),
                "hybrid_stage": row.get("hybrid_stage", ""),
                "hybrid_prefix_selected": row.get("hybrid_prefix_selected", ""),
                "hybrid_prefix_top_k": row.get("hybrid_prefix_top_k", ""),
                "hybrid_prefix_cluster_key": row.get("hybrid_prefix_cluster_key", ""),
                "hybrid_fill_cluster_key": row.get("hybrid_fill_cluster_key", ""),
                "quality_gate_threshold": row.get("quality_gate_threshold", ""),
                "quality_gate_pass": row.get("quality_gate_pass", ""),
                "quality_gate_old_novelty_score": row.get("quality_gate_old_novelty_score", ""),
                "quality_gate_source_cap": row.get("quality_gate_source_cap", ""),
                "quality_gate_source_key": row.get("quality_gate_source_key", ""),
                "quality_gate_source_id": row.get("quality_gate_source_id", ""),
                "source_cap_count_before_selection": row.get("source_cap_count_before_selection", ""),
                "was_source_cap_active": bool(row.get("was_source_cap_active", False)),
                "was_selected_by_source_cap_fallback": bool(row.get("was_selected_by_source_cap_fallback", False)),
                "quality_score": _safe_float(row.get("quality_score", 1.0)),
                "reason_code": row.get("reason_code", ""),
                "label": row.get("label", ""),
                "is_corruption": row.get("is_corruption", ""),
            }
        )
    return trace_rows


def _submission_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    submission_rows = []
    previous_score: float | None = None
    for row in rows:
        raw_score = _safe_float(row.get("rerank_score", row["final_score"]))
        score = raw_score
        if previous_score is not None and score >= previous_score:
            score = previous_score - 1.0e-12
        previous_score = score
        submission_rows.append(
            {
                "worker_id": row["worker_id"],
                "rank": row["rank"],
                "score": score,
                "quality_score": row.get("quality_score", 1.0),
                "reason_code": row.get("reason_code", ""),
            }
        )
    return submission_rows


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    row_list = [dict(row) for row in rows]
    fieldnames = sorted({key for row in row_list for key in row})
    _write_csv(path, fieldnames, row_list)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
