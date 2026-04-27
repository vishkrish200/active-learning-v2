from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from marginal_value.eval.encoder_eval import cosine_knn
from marginal_value.indexing.knn_features import normalize_rows


RARE_TEMPORAL_GRAMMAR_COMPONENT_THRESHOLD = 0.75
RARE_TEMPORAL_PROMOTION_DELTA_THRESHOLD = 0.08
RARE_TEMPORAL_MIN_DENSITY_SCORE = 0.35


def window_mean_std_embedding(windows: np.ndarray) -> np.ndarray:
    """Pool window-level features into the best Phase A baseline representation."""
    values = np.asarray(windows, dtype=float)
    if values.ndim == 2:
        return np.concatenate([np.nanmean(values, axis=0), np.nanstd(values, axis=0)])
    if values.ndim == 3:
        return np.concatenate([np.nanmean(values, axis=1), np.nanstd(values, axis=1)], axis=1)
    raise ValueError("windows must be a 2D sample array or 3D batch array")


def temporal_order_embedding(windows: np.ndarray, *, n_segments: int = 3) -> np.ndarray:
    values = _finite_2d(windows)
    segments = np.array_split(values, max(1, n_segments), axis=0)
    parts: list[np.ndarray] = []
    for segment in segments:
        if len(segment) == 0:
            parts.extend([np.zeros(values.shape[1]), np.zeros(values.shape[1])])
        else:
            parts.extend([np.mean(segment, axis=0), np.std(segment, axis=0)])
    while len(parts) < 2 * n_segments:
        parts.extend([np.zeros(values.shape[1]), np.zeros(values.shape[1])])
    if len(values) > 1:
        diffs = np.diff(values, axis=0)
        segment_size = max(1, len(values) // max(1, n_segments))
        parts.extend(
            [
                np.mean(np.abs(diffs), axis=0),
                np.max(np.abs(diffs), axis=0),
                np.mean(values[-segment_size:], axis=0) - np.mean(values[:segment_size], axis=0),
            ]
        )
    else:
        parts.extend([np.zeros(values.shape[1]), np.zeros(values.shape[1]), np.zeros(values.shape[1])])
    return np.nan_to_num(np.concatenate(parts), nan=0.0, posinf=0.0, neginf=0.0)


def raw_shape_stats_embedding(samples: np.ndarray, *, sample_rate: float = 30.0) -> np.ndarray:
    values = _finite_2d(samples)[:, :6]
    centered = values - np.mean(values, axis=0, keepdims=True)
    channel_parts = [
        np.mean(values, axis=0),
        np.std(values, axis=0),
        np.min(values, axis=0),
        np.max(values, axis=0),
        np.percentile(values, 25, axis=0),
        np.percentile(values, 75, axis=0),
        np.mean(values * values, axis=0),
    ]
    if len(values) > 1:
        diffs = np.diff(values, axis=0)
        abs_diffs = np.abs(diffs)
        diff_parts = [
            np.mean(abs_diffs, axis=0),
            np.std(diffs, axis=0),
            np.max(abs_diffs, axis=0),
            np.percentile(abs_diffs, 95, axis=0),
            np.mean(diffs * diffs, axis=0),
        ]
    else:
        diff_parts = [np.zeros(values.shape[1]) for _ in range(5)]

    spectral_parts = _spectral_band_energy(centered, sample_rate=sample_rate)
    autocorr_parts = _autocorrelation_features(centered, lags=(1, 3, 10))
    acc_energy = np.mean(values[:, :3] * values[:, :3], axis=0)
    gyro_energy = np.mean(values[:, 3:6] * values[:, 3:6], axis=0)
    axis_parts = [
        acc_energy / max(float(np.sum(acc_energy)), 1.0e-12),
        gyro_energy / max(float(np.sum(gyro_energy)), 1.0e-12),
    ]
    return np.nan_to_num(
        np.concatenate([*channel_parts, *diff_parts, spectral_parts, autocorr_parts, *axis_parts]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _spectral_band_energy(values: np.ndarray, *, sample_rate: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(values.shape[1] * 4, dtype=float)
    freqs = np.fft.rfftfreq(len(values), d=1.0 / max(float(sample_rate), 1.0e-6))
    power = np.abs(np.fft.rfft(values, axis=0)) ** 2
    total = np.sum(power, axis=0) + 1.0e-12
    nyquist = max(float(sample_rate) / 2.0, 1.0)
    bands = [(0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, nyquist + 1.0e-6)]
    parts = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            parts.append(np.zeros(values.shape[1], dtype=float))
        else:
            parts.append(np.sum(power[mask], axis=0) / total)
    return np.concatenate(parts)


def _autocorrelation_features(values: np.ndarray, *, lags: Sequence[int]) -> np.ndarray:
    parts = []
    denom = np.sum(values * values, axis=0) + 1.0e-12
    for lag in lags:
        if len(values) <= lag:
            parts.append(np.zeros(values.shape[1], dtype=float))
        else:
            parts.append(np.sum(values[:-lag] * values[lag:], axis=0) / denom)
    return np.concatenate(parts)


def _finite_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array.")
    if array.shape[0] == 0:
        raise ValueError("Expected at least one row.")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def minmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array, dtype=float)
    lo = float(np.min(array[finite]))
    hi = float(np.max(array[finite]))
    span = hi - lo
    if span < 1.0e-12:
        return np.zeros_like(array, dtype=float)
    scaled = (array - lo) / span
    return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)


def combine_novelty_density(
    novelty: np.ndarray,
    density: np.ndarray,
    *,
    novelty_weight: float = 0.75,
) -> np.ndarray:
    if not 0.0 <= novelty_weight <= 1.0:
        raise ValueError("novelty_weight must be in [0, 1]")
    novelty_score = minmax(np.asarray(novelty, dtype=float))
    density_score = minmax(np.asarray(density, dtype=float))
    if novelty_score.shape != density_score.shape:
        raise ValueError("novelty and density must have the same shape")
    return novelty_weight * novelty_score + (1.0 - novelty_weight) * density_score


def old_knn_novelty(
    support_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    *,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = cosine_knn(support_embeddings, candidate_embeddings, k=k)
    return np.mean(distances, axis=1), indices


def batch_density(embeddings: np.ndarray, *, k: int = 10) -> np.ndarray:
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    if len(vectors) <= 1:
        return np.zeros(len(vectors), dtype=float)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    k_eff = min(max(int(k), 1), len(vectors) - 1)
    order = np.argpartition(-similarities, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(len(vectors))[:, None]
    neighbor_sims = similarities[rows, order]
    return np.mean(np.maximum(neighbor_sims, 0.0), axis=1)


def compute_batch_clusters(embeddings: np.ndarray, *, similarity_threshold: float = 0.985) -> np.ndarray:
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1]")
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    n_rows = len(vectors)
    if n_rows == 0:
        return np.asarray([], dtype=int)
    similarities = vectors @ vectors.T
    visited = np.zeros(n_rows, dtype=bool)
    cluster_ids = np.full(n_rows, -1, dtype=int)
    cluster_id = 0
    for start in range(n_rows):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        cluster_ids[start] = cluster_id
        while stack:
            idx = stack.pop()
            neighbors = np.flatnonzero((similarities[idx] >= similarity_threshold) & ~visited)
            for neighbor in neighbors:
                visited[neighbor] = True
                cluster_ids[neighbor] = cluster_id
                stack.append(int(neighbor))
        cluster_id += 1
    return cluster_ids


def annotate_cluster_features(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
) -> list[dict[str, object]]:
    annotated = [dict(row) for row in rows]
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    ids = np.asarray(cluster_ids, dtype=int)
    if len(annotated) != len(vectors) or len(annotated) != len(ids):
        raise ValueError("rows, embeddings, and cluster_ids must have the same length")

    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(ids):
        cluster_to_indices[int(cluster_id)].append(idx)

    for cluster_id, indices in cluster_to_indices.items():
        cluster_vectors = vectors[indices]
        if len(indices) == 1:
            medoid_idx = indices[0]
        else:
            sims = cluster_vectors @ cluster_vectors.T
            medoid_idx = indices[int(np.argmax(np.mean(sims, axis=1)))]
        medoid = vectors[medoid_idx]
        cluster_size = len(indices)
        for idx in indices:
            annotated[idx]["new_cluster_id"] = int(cluster_id)
            annotated[idx]["new_cluster_size"] = int(cluster_size)
            annotated[idx]["is_singleton"] = bool(cluster_size == 1)
            annotated[idx]["distance_to_new_cluster_medoid"] = float(1.0 - vectors[idx] @ medoid)
    return annotated


def split_large_clusters(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray | None = None,
    *,
    max_cluster_size: int = 300,
    target_subcluster_size: int = 75,
    score_columns: Sequence[str] = (
        "grammar_score_component",
        "grammar_score",
        "old_novelty_score",
        "new_density_score",
        "distance_to_new_cluster_medoid",
        "final_score",
    ),
    split_method: str = "feature_kmeans",
    score_feature_weight: float = 0.0,
    kmeans_iterations: int = 16,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Deterministically split coarse mega-clusters before diversity reranking.

    When embeddings are provided, the split is based on the same feature-space
    geometry used by ranking. Scores may be appended as weak auxiliary features,
    but the default deliberately ignores them so diversity metrics cannot be
    inflated by score round-robin assignment.
    """
    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be positive.")
    if target_subcluster_size < 1:
        raise ValueError("target_subcluster_size must be positive.")
    if split_method not in {"feature_kmeans", "score_round_robin"}:
        raise ValueError("split_method must be 'feature_kmeans' or 'score_round_robin'.")
    if score_feature_weight < 0.0:
        raise ValueError("score_feature_weight must be non-negative.")
    if kmeans_iterations < 1:
        raise ValueError("kmeans_iterations must be positive.")
    candidates = [dict(row) for row in rows]
    if not candidates:
        return [], {
            "enabled": True,
            "n_rows": 0,
            "n_clusters_before": 0,
            "n_clusters_after": 0,
            "n_split_parent_clusters": 0,
            "max_cluster_size_before": 0,
            "max_cluster_size_after": 0,
            "split_method": split_method,
        }
    if any("new_cluster_id" not in row for row in candidates):
        raise ValueError("split_large_clusters requires rows annotated with new_cluster_id.")
    vectors: np.ndarray | None = None
    if embeddings is not None:
        vectors = normalize_rows(np.asarray(embeddings, dtype=float))
        if len(vectors) != len(candidates):
            raise ValueError("rows and embeddings must have the same length.")

    by_cluster: defaultdict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(candidates):
        by_cluster[int(row["new_cluster_id"])].append(idx)

    next_cluster_id = max(by_cluster) + 1
    split_parent_count = 0
    feature_split_count = 0
    fallback_split_count = 0
    for parent_cluster_id, indices in sorted(by_cluster.items()):
        if len(indices) <= max_cluster_size:
            for idx in indices:
                candidates[idx]["new_cluster_parent_id"] = int(parent_cluster_id)
                candidates[idx]["large_cluster_split_applied"] = False
                candidates[idx]["new_cluster_subcluster_id"] = 0
                candidates[idx]["large_cluster_split_strategy"] = "not_split"
            continue

        split_parent_count += 1
        n_subclusters = max(2, int(np.ceil(len(indices) / target_subcluster_size)))
        if split_method == "feature_kmeans" and vectors is not None:
            labels = _feature_subcluster_labels(
                candidates,
                vectors,
                indices,
                n_subclusters=n_subclusters,
                max_subcluster_size=target_subcluster_size,
                score_columns=score_columns,
                score_feature_weight=score_feature_weight,
                kmeans_iterations=kmeans_iterations,
            )
            feature_split_count += 1
            split_strategy = "feature_kmeans"
            indexed_labels = list(zip(indices, labels, strict=True))
        else:
            ranked_indices = sorted(
                indices,
                key=lambda idx: (
                    -_composite_split_score(candidates[idx], score_columns),
                    str(candidates[idx].get("sample_id", candidates[idx].get("worker_id", ""))),
                ),
            )
            labels = [order_idx % n_subclusters for order_idx, _idx in enumerate(ranked_indices)]
            indexed_labels = list(zip(ranked_indices, labels, strict=True))
            fallback_split_count += 1
            split_strategy = "score_round_robin_fallback"
        subcluster_ids = [next_cluster_id + offset for offset in range(n_subclusters)]
        next_cluster_id += n_subclusters
        for idx, subcluster_idx in indexed_labels:
            candidates[idx]["new_cluster_parent_id"] = int(parent_cluster_id)
            candidates[idx]["new_cluster_id"] = int(subcluster_ids[subcluster_idx])
            candidates[idx]["new_cluster_subcluster_id"] = int(subcluster_idx)
            candidates[idx]["large_cluster_split_applied"] = True
            candidates[idx]["large_cluster_split_strategy"] = split_strategy

    counts: defaultdict[int, int] = defaultdict(int)
    for row in candidates:
        counts[int(row["new_cluster_id"])] += 1
    for row in candidates:
        size = int(counts[int(row["new_cluster_id"])])
        row["new_cluster_size"] = size
        row["is_singleton"] = bool(size == 1)
    if vectors is not None:
        _refresh_split_medoid_distances(candidates, vectors)

    before_sizes = [len(indices) for indices in by_cluster.values()]
    after_sizes = list(counts.values())
    summary = {
        "enabled": True,
        "n_rows": len(candidates),
        "n_clusters_before": len(before_sizes),
        "n_clusters_after": len(after_sizes),
        "n_split_parent_clusters": split_parent_count,
        "max_cluster_size_before": int(max(before_sizes)) if before_sizes else 0,
        "max_cluster_size_after": int(max(after_sizes)) if after_sizes else 0,
        "max_cluster_size": int(max_cluster_size),
        "target_subcluster_size": int(target_subcluster_size),
        "score_columns": list(score_columns),
        "split_method": split_method,
        "feature_split_parent_clusters": feature_split_count,
        "fallback_split_parent_clusters": fallback_split_count,
        "score_feature_weight": float(score_feature_weight),
        "kmeans_iterations": int(kmeans_iterations),
        "embedding_features_used": bool(vectors is not None),
    }
    return candidates, summary


def apply_grammar_score_promotion(
    rows: Iterable[dict[str, object]],
    *,
    score_variant: str,
    score_weight: float,
    min_quality: float,
    min_new_density_score: float,
) -> list[dict[str, object]]:
    if score_variant not in {"grammar_surprisal_mix", "grammar_phrase_mix", "token_nll_p95", "quality_gated_grammar"}:
        raise ValueError("score_variant must be one of grammar_surprisal_mix, grammar_phrase_mix, token_nll_p95, quality_gated_grammar.")
    max_score_weight = 1.0 if score_variant == "quality_gated_grammar" else 0.35
    if not 0.0 <= score_weight <= max_score_weight:
        raise ValueError(f"score_weight must be in [0, {max_score_weight}].")
    if not 0.0 <= min_quality <= 1.0:
        raise ValueError("min_quality must be in [0, 1].")
    if not 0.0 <= min_new_density_score <= 1.0:
        raise ValueError("min_new_density_score must be in [0, 1].")

    promoted = [dict(row) for row in rows]
    grammar_scores = _grammar_variant_scores(promoted, score_variant)
    for row, grammar_score in zip(promoted, grammar_scores):
        quality = float(row.get("quality_score", 1.0))
        support = float(row.get("new_density_score", row.get("new_batch_density", 0.0)))
        phase_a_ranker_score = float(row.get("ranker_score", 0.0))
        phase_a_final_score = float(row.get("final_score", quality * phase_a_ranker_score))
        has_grammar = _truthy(row.get("grammar_feature_present", False))
        row["phase_a_ranker_score"] = phase_a_ranker_score
        row["phase_a_final_score"] = phase_a_final_score
        row["grammar_score"] = float(grammar_score)
        row["grammar_score_variant"] = score_variant
        row["grammar_score_weight"] = float(score_weight)
        if not has_grammar:
            row["grammar_score_component"] = 0.0
            row["grammar_promotion_delta"] = 0.0
            row["grammar_promotion_applied"] = False
            continue

        gate = 1.0 if quality >= min_quality and support >= min_new_density_score else 0.0
        if score_variant == "quality_gated_grammar":
            grammar_component = float(grammar_score * gate)
            ranker_score = (1.0 - score_weight) * phase_a_ranker_score + score_weight * grammar_component
        else:
            grammar_component = float(grammar_score * support * gate)
            ranker_score = (1.0 - score_weight) * phase_a_ranker_score + score_weight * grammar_component
        final_score = float(np.clip(quality * ranker_score, 0.0, 1.0))
        row["grammar_score_component"] = grammar_component
        row["grammar_support_gate"] = float(gate)
        row["grammar_promotion_delta"] = float(final_score - phase_a_final_score)
        row["grammar_promotion_applied"] = True
        row["ranker_score"] = float(np.clip(ranker_score, 0.0, 1.0))
        row["final_score"] = final_score

    reasons = build_reason_codes(promoted)
    for row, reason in zip(promoted, reasons):
        row["reason_code"] = reason
    return promoted


def apply_stationary_singleton_guard(
    rows: Iterable[dict[str, object]],
    *,
    stationary_threshold: float = 0.90,
    max_new_density_score: float = 0.35,
    min_grammar_score: float = 0.85,
    penalty_multiplier: float = 0.35,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Downweight high-stationary singleton novelty unless another signal supports it."""

    if not 0.0 <= stationary_threshold <= 1.0:
        raise ValueError("stationary_threshold must be in [0, 1].")
    if not 0.0 <= max_new_density_score <= 1.0:
        raise ValueError("max_new_density_score must be in [0, 1].")
    if not 0.0 <= min_grammar_score <= 1.0:
        raise ValueError("min_grammar_score must be in [0, 1].")
    if not 0.0 <= penalty_multiplier <= 1.0:
        raise ValueError("penalty_multiplier must be in [0, 1].")

    guarded: list[dict[str, object]] = []
    applied_count = 0
    for row in rows:
        output = dict(row)
        novelty = float(output.get("old_novelty_score", output.get("novelty_score", 0.0)))
        density = float(output.get("new_density_score", output.get("density_score", 0.0)))
        stationary = float(output.get("stationary_fraction", 0.0))
        grammar = float(output.get("grammar_score_component", output.get("grammar_score", 0.0)))
        reason = str(output.get("reason_code", ""))
        is_singleton_novelty = reason == "HIGH_NOVELTY_SINGLETON" or (novelty >= 0.65 and density < max_new_density_score)
        has_support = density >= max_new_density_score
        has_grammar_support = grammar >= min_grammar_score
        should_penalize = (
            is_singleton_novelty
            and stationary >= stationary_threshold
            and not has_support
            and not has_grammar_support
        )
        output["stationary_singleton_guard_applied"] = bool(should_penalize)
        output["stationary_singleton_guard_stationary_threshold"] = float(stationary_threshold)
        output["stationary_singleton_guard_max_new_density_score"] = float(max_new_density_score)
        output["stationary_singleton_guard_min_grammar_score"] = float(min_grammar_score)
        output["stationary_singleton_guard_penalty_multiplier"] = float(penalty_multiplier)
        output["pre_stationary_guard_final_score"] = float(output.get("final_score", 0.0))
        output["pre_stationary_guard_ranker_score"] = float(output.get("ranker_score", 0.0))
        if should_penalize:
            applied_count += 1
            quality = float(output.get("quality_score", 1.0))
            output["ranker_score"] = float(np.clip(float(output.get("ranker_score", 0.0)) * penalty_multiplier, 0.0, 1.0))
            output["final_score"] = float(np.clip(quality * output["ranker_score"], 0.0, 1.0))
            output["stationary_singleton_guard_penalty"] = float(
                float(output["pre_stationary_guard_final_score"]) - float(output["final_score"])
            )
        else:
            output["stationary_singleton_guard_penalty"] = 0.0
        guarded.append(output)

    reasons = build_reason_codes(guarded)
    for row, reason in zip(guarded, reasons):
        row["reason_code"] = reason

    summary = {
        "enabled": True,
        "applied": applied_count > 0,
        "applied_count": applied_count,
        "n_rows": len(guarded),
        "stationary_threshold": float(stationary_threshold),
        "max_new_density_score": float(max_new_density_score),
        "min_grammar_score": float(min_grammar_score),
        "penalty_multiplier": float(penalty_multiplier),
    }
    return guarded, summary


def build_reason_codes(rows: Iterable[dict[str, object]]) -> list[str]:
    reasons: list[str] = []
    for row in rows:
        quality = float(row.get("quality_score", 1.0))
        novelty = float(row.get("old_novelty_score", row.get("novelty_score", 0.0)))
        density = float(row.get("new_density_score", row.get("density_score", 0.0)))
        grammar_component = float(row.get("grammar_score_component", row.get("grammar_score", 0.0)))
        grammar_delta = float(row.get("grammar_promotion_delta", 0.0))
        if quality < 0.45:
            reasons.append("LOW_QUALITY")
        elif novelty >= 0.65 and density >= 0.45:
            reasons.append("COHESIVE_NEW_WORKFLOW")
        elif novelty >= 0.65:
            reasons.append("HIGH_NOVELTY_SINGLETON")
        elif (
            grammar_component >= RARE_TEMPORAL_GRAMMAR_COMPONENT_THRESHOLD
            and grammar_delta >= RARE_TEMPORAL_PROMOTION_DELTA_THRESHOLD
            and density >= RARE_TEMPORAL_MIN_DENSITY_SCORE
        ):
            reasons.append("RARE_TEMPORAL_COMPOSITION")
        elif novelty < 0.35 and density >= 0.65:
            reasons.append("REDUNDANT_KNOWN_WORKFLOW")
        else:
            reasons.append("RARE_MOTION_PRIMITIVES")
    return reasons


def build_scored_rows(
    *,
    sample_ids: Sequence[str],
    embeddings: np.ndarray,
    old_knn_distance: np.ndarray,
    new_density: np.ndarray,
    quality_scores: np.ndarray | None = None,
    novelty_weight: float = 0.75,
) -> list[dict[str, object]]:
    if quality_scores is None:
        quality_scores = np.ones(len(sample_ids), dtype=float)
    novelty_score = minmax(old_knn_distance)
    density_score = minmax(new_density)
    ranker_score = combine_novelty_density(old_knn_distance, new_density, novelty_weight=novelty_weight)
    final_score = np.asarray(quality_scores, dtype=float) * ranker_score
    rows: list[dict[str, object]] = []
    for idx, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "worker_id": sample_id,
                "sample_id": sample_id,
                "quality_score": float(quality_scores[idx]),
                "old_knn_distance": float(old_knn_distance[idx]),
                "new_batch_density": float(new_density[idx]),
                "old_novelty_score": float(novelty_score[idx]),
                "new_density_score": float(density_score[idx]),
                "ranker_score": float(ranker_score[idx]),
                "final_score": float(final_score[idx]),
            }
        )
    reasons = build_reason_codes(rows)
    for row, reason in zip(rows, reasons):
        row["reason_code"] = reason
    return rows


def mmr_rank_rows(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray,
    *,
    lambda_redundancy: float = 0.25,
) -> list[dict[str, object]]:
    reranked = _diversity_rerank(rows, embeddings, lambda_redundancy=lambda_redundancy)
    for rank, row in enumerate(reranked, start=1):
        row["rank"] = rank
    return reranked


def cluster_aware_rank_rows(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray,
    *,
    lambda_redundancy: float = 0.25,
    cluster_bonus_weight: float = 0.25,
) -> list[dict[str, object]]:
    candidates = [dict(row) for row in rows]
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    if len(candidates) != len(vectors):
        raise ValueError("rows and embeddings must have the same length")
    if not candidates:
        return []

    if any("new_cluster_id" not in row for row in candidates):
        cluster_ids = compute_batch_clusters(vectors)
        candidates = annotate_cluster_features(candidates, vectors, cluster_ids)

    similarities = vectors @ vectors.T
    base_scores = np.asarray([float(row.get("final_score", 0.0)) for row in candidates], dtype=float)
    redundancy = np.zeros(len(candidates), dtype=float)
    selected: list[int] = []
    remaining = np.ones(len(candidates), dtype=bool)
    seen_clusters: set[int] = set()
    cluster_counts: defaultdict[int, int] = defaultdict(int)
    while len(selected) < len(candidates):
        cluster_bonus = np.asarray(
            [
                cluster_bonus_weight if int(row["new_cluster_id"]) not in seen_clusters else 0.0
                for row in candidates
            ],
            dtype=float,
        )
        scores = base_scores + cluster_bonus - lambda_redundancy * np.maximum(redundancy, 0.0)
        scores[~remaining] = -np.inf
        best_idx = int(np.argmax(scores))
        if not np.isfinite(scores[best_idx]):
            break
        cluster_id = int(candidates[best_idx]["new_cluster_id"])
        count_before = int(cluster_counts[cluster_id])
        candidates[best_idx]["redundancy_penalty"] = float(max(redundancy[best_idx], 0.0))
        candidates[best_idx]["cluster_bonus"] = float(cluster_bonus[best_idx])
        candidates[best_idx]["cluster_selection_score"] = float(scores[best_idx])
        candidates[best_idx]["cluster_round"] = count_before
        candidates[best_idx]["cluster_count_before_selection"] = count_before
        candidates[best_idx]["was_cluster_cap_active"] = False
        candidates[best_idx]["was_selected_by_fallback"] = False
        selected.append(best_idx)
        remaining[best_idx] = False
        seen_clusters.add(cluster_id)
        cluster_counts[cluster_id] += 1
        redundancy = np.maximum(redundancy, similarities[:, best_idx])

    ranked = [candidates[idx] for idx in selected]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["reranker"] = "cluster_aware"
        row["rerank_score"] = float(row["cluster_selection_score"])
    return ranked


def cluster_cap_rank_rows(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray,
    *,
    lambda_redundancy: float = 0.25,
    cluster_bonus_weight: float = 0.25,
    cluster_cap_top_k: int = 200,
    cluster_max_per_cluster: int = 3,
    cluster_key: str = "new_cluster_id",
    cluster_cap_min_quality: float = 0.0,
) -> list[dict[str, object]]:
    if cluster_cap_top_k < 1:
        raise ValueError("cluster_cap_top_k must be positive.")
    if cluster_max_per_cluster < 1:
        raise ValueError("cluster_max_per_cluster must be positive.")
    if not cluster_key:
        raise ValueError("cluster_key must be non-empty.")
    if not 0.0 <= cluster_cap_min_quality <= 1.0:
        raise ValueError("cluster_cap_min_quality must be in [0, 1].")
    candidates = [dict(row) for row in rows]
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    if len(candidates) != len(vectors):
        raise ValueError("rows and embeddings must have the same length")
    if not candidates:
        return []

    if any("new_cluster_id" not in row for row in candidates):
        cluster_ids = compute_batch_clusters(vectors)
        candidates = annotate_cluster_features(candidates, vectors, cluster_ids)
    if cluster_key == "new_cluster_parent_id":
        for row in candidates:
            row.setdefault("new_cluster_parent_id", row.get("new_cluster_id", -1))
    if any(cluster_key not in row for row in candidates):
        raise ValueError(f"cluster_cap_rank_rows requires rows annotated with {cluster_key}.")

    similarities = vectors @ vectors.T
    base_scores = np.asarray([float(row.get("final_score", 0.0)) for row in candidates], dtype=float)
    redundancy = np.zeros(len(candidates), dtype=float)
    selected: list[int] = []
    remaining = np.ones(len(candidates), dtype=bool)
    seen_clusters: set[int] = set()
    cluster_counts: defaultdict[int, int] = defaultdict(int)
    while len(selected) < len(candidates):
        eligible = remaining.copy()
        cap_window_active = len(selected) < cluster_cap_top_k
        under_cap_remaining_count = 0
        selected_by_fallback = False
        cap_filter_active = False
        if cap_window_active:
            under_cap = np.asarray(
                [
                    cluster_counts[int(row[cluster_key])] < cluster_max_per_cluster
                    and float(row.get("quality_score", 1.0)) >= cluster_cap_min_quality
                    for row in candidates
                ],
                dtype=bool,
            )
            under_cap_remaining_count = int(np.sum(eligible & under_cap))
            if under_cap_remaining_count > 0:
                eligible &= under_cap
                cap_filter_active = True
            else:
                selected_by_fallback = True
        cluster_bonus = np.asarray(
            [
                cluster_bonus_weight if int(row[cluster_key]) not in seen_clusters else 0.0
                for row in candidates
            ],
            dtype=float,
        )
        scores = base_scores + cluster_bonus - lambda_redundancy * np.maximum(redundancy, 0.0)
        scores[~eligible] = -np.inf
        best_idx = int(np.argmax(scores))
        if not np.isfinite(scores[best_idx]):
            break
        cluster_id = int(candidates[best_idx][cluster_key])
        count_before = int(cluster_counts[cluster_id])
        candidates[best_idx]["redundancy_penalty"] = float(max(redundancy[best_idx], 0.0))
        candidates[best_idx]["cluster_bonus"] = float(cluster_bonus[best_idx])
        candidates[best_idx]["cluster_selection_score"] = float(scores[best_idx])
        candidates[best_idx]["cluster_round"] = count_before
        candidates[best_idx]["cluster_count_before_selection"] = count_before
        candidates[best_idx]["cluster_cap_key"] = cluster_key
        candidates[best_idx]["cluster_cap_cluster_id"] = cluster_id
        candidates[best_idx]["cluster_cap_min_quality"] = float(cluster_cap_min_quality)
        candidates[best_idx]["was_cluster_cap_active"] = bool(cap_filter_active)
        candidates[best_idx]["was_selected_by_fallback"] = bool(selected_by_fallback)
        candidates[best_idx]["eligible_under_cluster_cap_count"] = under_cap_remaining_count
        candidates[best_idx]["cluster_cap_top_k"] = int(cluster_cap_top_k)
        candidates[best_idx]["cluster_max_per_cluster"] = int(cluster_max_per_cluster)
        selected.append(best_idx)
        remaining[best_idx] = False
        seen_clusters.add(cluster_id)
        cluster_counts[cluster_id] += 1
        redundancy = np.maximum(redundancy, similarities[:, best_idx])

    ranked = [candidates[idx] for idx in selected]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["reranker"] = "cluster_cap"
        row["rerank_score"] = float(row["cluster_selection_score"])
    return ranked


def quality_gated_old_novelty_rank_rows(
    rows: Iterable[dict[str, object]],
    *,
    quality_threshold: float = 0.45,
    max_stationary_fraction: float | None = None,
    max_abs_value: float | None = None,
    source_cap: int | None = None,
    source_key: str = "source_group_id",
) -> list[dict[str, object]]:
    """Rank clean candidates by old-support novelty, with optional source cap."""
    if not 0.0 <= quality_threshold <= 1.0:
        raise ValueError("quality_threshold must be in [0, 1].")
    if max_stationary_fraction is not None and not 0.0 <= float(max_stationary_fraction) <= 1.0:
        raise ValueError("max_stationary_fraction must be in [0, 1] when provided.")
    if max_abs_value is not None and float(max_abs_value) < 0.0:
        raise ValueError("max_abs_value must be non-negative when provided.")
    if source_cap is not None and int(source_cap) <= 0:
        raise ValueError("source_cap must be positive when provided.")
    if source_cap is not None and not source_key:
        raise ValueError("source_key must be non-empty when source_cap is provided.")

    candidates = []
    for index, row in enumerate(rows):
        output = dict(row)
        quality = _row_float(output, "quality_score", 0.0)
        novelty = _row_float(output, "old_novelty_score", _row_float(output, "novelty_score", 0.0))
        quality_pass = bool(quality >= quality_threshold)
        physical_pass, physical_failures = _physical_validity_gate(
            output,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
        passed = bool(quality_pass and physical_pass)
        output["_qgate_source_index"] = index
        output["quality_threshold_pass"] = quality_pass
        output["quality_gate_threshold"] = float(quality_threshold)
        output["quality_gate_pass"] = passed
        output["physical_validity_pass"] = physical_pass
        output["physical_validity_failure_reasons"] = ",".join(physical_failures)
        output["physical_validity_max_stationary_fraction"] = (
            "" if max_stationary_fraction is None else float(max_stationary_fraction)
        )
        output["physical_validity_max_abs_value"] = "" if max_abs_value is None else float(max_abs_value)
        output["quality_gate_old_novelty_score"] = float(novelty if passed else 0.0)
        output["quality_gate_source_cap"] = "" if source_cap is None else int(source_cap)
        output["quality_gate_source_key"] = source_key if source_cap is not None else ""
        output["was_source_cap_active"] = False
        output["was_selected_by_source_cap_fallback"] = False
        output["source_cap_count_before_selection"] = 0
        candidates.append(output)

    ordered = sorted(candidates, key=_quality_gated_old_novelty_sort_key)
    if source_cap is None:
        ranked = ordered
        reranker = "quality_gated_old_novelty"
    else:
        ranked = _apply_source_cap_order(ordered, source_cap=int(source_cap), source_key=source_key)
        reranker = "quality_gated_old_novelty_sourcecap"

    for rank, row in enumerate(ranked, start=1):
        row.pop("_qgate_source_index", None)
        row["rank"] = rank
        row["reranker"] = reranker
        row["rerank_score"] = float(row.get("quality_gate_old_novelty_score", 0.0))
    return ranked


def _physical_validity_gate(
    row: dict[str, object],
    *,
    max_stationary_fraction: float | None,
    max_abs_value: float | None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if max_stationary_fraction is not None:
        stationary = _row_float(row, "stationary_fraction", 0.0)
        if stationary > float(max_stationary_fraction):
            failures.append("stationary_fraction")
    if max_abs_value is not None:
        observed = _row_float(row, "max_abs_value", _row_float(row, "max_abs", 0.0))
        if observed > float(max_abs_value):
            failures.append("max_abs_value")
    return len(failures) == 0, failures


def quality_only_rank_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [dict(row) for row in rows]
    ranked = sorted(
        candidates,
        key=lambda row: (
            -_row_float(row, "quality_score", 0.0),
            -_row_float(row, "old_novelty_score", _row_float(row, "novelty_score", 0.0)),
            str(row.get("sample_id", row.get("worker_id", ""))),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["reranker"] = "quality_only"
        row["rerank_score"] = _row_float(row, "quality_score", 0.0)
    return ranked


def tiered_cluster_cap_rank_rows(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray,
    *,
    cap_schedule: Sequence[dict[str, object]],
    cluster_key: str = "new_cluster_id",
    lambda_redundancy: float = 0.25,
    cluster_bonus_weight: float = 0.25,
    cluster_cap_min_quality: float = 0.0,
) -> list[dict[str, object]]:
    """Apply strict early cluster caps, then relax them at later rank cutoffs."""
    schedule = _validate_cap_schedule(cap_schedule)
    if not cluster_key:
        raise ValueError("cluster_key must be non-empty.")
    if not 0.0 <= cluster_cap_min_quality <= 1.0:
        raise ValueError("cluster_cap_min_quality must be in [0, 1].")
    candidates = [dict(row) for row in rows]
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    if len(candidates) != len(vectors):
        raise ValueError("rows and embeddings must have the same length")
    if not candidates:
        return []

    if any("new_cluster_id" not in row for row in candidates):
        cluster_ids = compute_batch_clusters(vectors)
        candidates = annotate_cluster_features(candidates, vectors, cluster_ids)
    if cluster_key == "new_cluster_parent_id":
        for row in candidates:
            row.setdefault("new_cluster_parent_id", row.get("new_cluster_id", -1))
    if any(cluster_key not in row for row in candidates):
        raise ValueError(f"tiered_cluster_cap_rank_rows requires rows annotated with {cluster_key}.")

    similarities = vectors @ vectors.T
    base_scores = np.asarray([float(row.get("final_score", 0.0)) for row in candidates], dtype=float)
    redundancy = np.zeros(len(candidates), dtype=float)
    selected: list[int] = []
    remaining = np.ones(len(candidates), dtype=bool)
    seen_clusters: set[int] = set()
    cluster_counts: defaultdict[int, int] = defaultdict(int)
    schedule_label = ";".join(f"{tier['top_k']}:{tier['max_per_cluster']}" for tier in schedule)
    while len(selected) < len(candidates):
        rank_position = len(selected) + 1
        tier = _cap_tier_for_rank(schedule, rank_position)
        eligible = remaining.copy()
        under_cap_remaining_count = 0
        selected_by_fallback = False
        cap_filter_active = False
        tier_top_k = 0
        tier_max_per_cluster = 0
        if tier is not None:
            tier_top_k = int(tier["top_k"])
            tier_max_per_cluster = int(tier["max_per_cluster"])
            under_cap = np.asarray(
                [
                    cluster_counts[int(row[cluster_key])] < tier_max_per_cluster
                    and float(row.get("quality_score", 1.0)) >= cluster_cap_min_quality
                    for row in candidates
                ],
                dtype=bool,
            )
            under_cap_remaining_count = int(np.sum(eligible & under_cap))
            if under_cap_remaining_count > 0:
                eligible &= under_cap
                cap_filter_active = True
            else:
                selected_by_fallback = True
        cluster_bonus = np.asarray(
            [
                cluster_bonus_weight if int(row[cluster_key]) not in seen_clusters else 0.0
                for row in candidates
            ],
            dtype=float,
        )
        scores = base_scores + cluster_bonus - lambda_redundancy * np.maximum(redundancy, 0.0)
        scores[~eligible] = -np.inf
        best_idx = int(np.argmax(scores))
        if not np.isfinite(scores[best_idx]):
            break
        cluster_id = int(candidates[best_idx][cluster_key])
        count_before = int(cluster_counts[cluster_id])
        candidates[best_idx]["redundancy_penalty"] = float(max(redundancy[best_idx], 0.0))
        candidates[best_idx]["cluster_bonus"] = float(cluster_bonus[best_idx])
        candidates[best_idx]["cluster_selection_score"] = float(scores[best_idx])
        candidates[best_idx]["cluster_round"] = count_before
        candidates[best_idx]["cluster_count_before_selection"] = count_before
        candidates[best_idx]["cluster_cap_key"] = cluster_key
        candidates[best_idx]["cluster_cap_cluster_id"] = cluster_id
        candidates[best_idx]["cluster_cap_min_quality"] = float(cluster_cap_min_quality)
        candidates[best_idx]["was_cluster_cap_active"] = bool(cap_filter_active)
        candidates[best_idx]["was_selected_by_fallback"] = bool(selected_by_fallback)
        candidates[best_idx]["eligible_under_cluster_cap_count"] = under_cap_remaining_count
        candidates[best_idx]["cluster_cap_top_k"] = tier_top_k
        candidates[best_idx]["cluster_max_per_cluster"] = tier_max_per_cluster
        candidates[best_idx]["cluster_cap_schedule"] = schedule_label
        selected.append(best_idx)
        remaining[best_idx] = False
        seen_clusters.add(cluster_id)
        cluster_counts[cluster_id] += 1
        redundancy = np.maximum(redundancy, similarities[:, best_idx])

    ranked = [candidates[idx] for idx in selected]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["reranker"] = "tiered_cluster_cap"
        row["rerank_score"] = float(row["cluster_selection_score"])
    return ranked


def parent_prefix_cluster_cap_rank_rows(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray,
    *,
    prefix_top_k: int = 75,
    prefix_cluster_key: str = "new_cluster_parent_id",
    prefix_max_per_cluster: int = 8,
    fill_cluster_key: str = "new_cluster_id",
    fill_max_per_cluster: int = 8,
    cluster_cap_top_k: int = 200,
    lambda_redundancy: float = 0.25,
    cluster_bonus_weight: float = 0.0,
    cluster_cap_min_quality: float = 0.0,
) -> list[dict[str, object]]:
    """Use parent-cluster diversity for the first ranks, then fill by child-cluster cap."""
    if prefix_top_k < 1:
        raise ValueError("prefix_top_k must be positive.")
    prepared = [dict(row, _hybrid_source_index=idx) for idx, row in enumerate(rows)]
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    if len(prepared) != len(vectors):
        raise ValueError("rows and embeddings must have the same length")
    if not prepared:
        return []

    prefix_ranked = cluster_cap_rank_rows(
        prepared,
        vectors,
        lambda_redundancy=lambda_redundancy,
        cluster_bonus_weight=cluster_bonus_weight,
        cluster_cap_top_k=prefix_top_k,
        cluster_max_per_cluster=prefix_max_per_cluster,
        cluster_key=prefix_cluster_key,
        cluster_cap_min_quality=cluster_cap_min_quality,
    )
    fill_ranked = cluster_cap_rank_rows(
        prepared,
        vectors,
        lambda_redundancy=lambda_redundancy,
        cluster_bonus_weight=cluster_bonus_weight,
        cluster_cap_top_k=cluster_cap_top_k,
        cluster_max_per_cluster=fill_max_per_cluster,
        cluster_key=fill_cluster_key,
        cluster_cap_min_quality=cluster_cap_min_quality,
    )

    selected: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    for row in prefix_ranked[:prefix_top_k]:
        idx = int(row["_hybrid_source_index"])
        selected.append(dict(row, hybrid_stage="prefix", hybrid_prefix_selected=True))
        seen_indices.add(idx)
    for row in fill_ranked:
        idx = int(row["_hybrid_source_index"])
        if idx in seen_indices:
            continue
        selected.append(dict(row, hybrid_stage="fill", hybrid_prefix_selected=False))
        seen_indices.add(idx)

    for rank, row in enumerate(selected, start=1):
        row.pop("_hybrid_source_index", None)
        row["rank"] = rank
        row["reranker"] = "parent_prefix_cluster_cap"
        row["hybrid_prefix_top_k"] = int(prefix_top_k)
        row["hybrid_prefix_cluster_key"] = prefix_cluster_key
        row["hybrid_fill_cluster_key"] = fill_cluster_key
        row["hybrid_prefix_max_per_cluster"] = int(prefix_max_per_cluster)
        row["hybrid_fill_max_per_cluster"] = int(fill_max_per_cluster)
        row["rerank_score"] = float(row.get("cluster_selection_score", row.get("final_score", 0.0)))
    return selected


def _validate_cap_schedule(cap_schedule: Sequence[dict[str, object]]) -> list[dict[str, int]]:
    if not cap_schedule:
        raise ValueError("cap_schedule must contain at least one tier.")
    schedule: list[dict[str, int]] = []
    previous_top_k = 0
    for tier in cap_schedule:
        top_k = int(tier.get("top_k", 0))
        max_per_cluster = int(tier.get("max_per_cluster", 0))
        if top_k <= 0:
            raise ValueError("cap_schedule tiers must have positive top_k.")
        if max_per_cluster <= 0:
            raise ValueError("cap_schedule tiers must have positive max_per_cluster.")
        if top_k <= previous_top_k:
            raise ValueError("cap_schedule top_k values must be strictly increasing.")
        schedule.append({"top_k": top_k, "max_per_cluster": max_per_cluster})
        previous_top_k = top_k
    return schedule


def _cap_tier_for_rank(schedule: Sequence[dict[str, int]], rank_position: int) -> dict[str, int] | None:
    for tier in schedule:
        if rank_position <= int(tier["top_k"]):
            return dict(tier)
    return None


def _apply_source_cap_order(
    ordered: Sequence[dict[str, object]],
    *,
    source_cap: int,
    source_key: str,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    source_counts: defaultdict[str, int] = defaultdict(int)
    for row in ordered:
        output = dict(row)
        source = _source_cap_identifier(output, source_key)
        output["quality_gate_source_id"] = source
        output["quality_gate_source_cap"] = int(source_cap)
        output["quality_gate_source_key"] = source_key
        output["source_cap_count_before_selection"] = int(source_counts[source])
        if bool(output.get("quality_gate_pass", False)) and source_counts[source] < source_cap:
            output["was_source_cap_active"] = True
            selected.append(output)
            source_counts[source] += 1
        else:
            output["was_selected_by_source_cap_fallback"] = bool(output.get("quality_gate_pass", False))
            skipped.append(output)
    return [*selected, *skipped]


def _source_cap_identifier(row: dict[str, object], source_key: str) -> str:
    if source_key == "new_cluster_parent_id" and source_key not in row and "new_cluster_id" in row:
        row["new_cluster_parent_id"] = row["new_cluster_id"]
    if source_key not in row:
        raise ValueError(f"source_key {source_key} is missing from a ranked row.")
    return str(row[source_key])


def _quality_gated_old_novelty_sort_key(row: dict[str, object]) -> tuple[int, int, float, float, str]:
    passed = 1 if bool(row.get("quality_gate_pass", False)) else 0
    quality_pass = 1 if bool(row.get("quality_threshold_pass", row.get("quality_gate_pass", False))) else 0
    novelty = _row_float(row, "old_novelty_score", _row_float(row, "novelty_score", 0.0))
    quality = _row_float(row, "quality_score", 0.0)
    sample_id = str(row.get("sample_id", row.get("worker_id", row.get("_qgate_source_index", ""))))
    return (-passed, -quality_pass, -novelty, -quality, sample_id)


def _diversity_rerank(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray,
    *,
    lambda_redundancy: float,
) -> list[dict[str, object]]:
    candidates = [dict(row) for row in rows]
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    if len(candidates) != len(vectors):
        raise ValueError("rows and embeddings must have the same length")

    remaining = set(range(len(candidates)))
    selected: list[int] = []
    while remaining:
        best_idx = None
        best_score = -float("inf")
        best_redundancy = 0.0
        for idx in remaining:
            base_score = float(candidates[idx].get("final_score", 0.0))
            redundancy = 0.0
            if selected:
                redundancy = float(np.max(vectors[idx] @ vectors[selected].T))
            mmr_score = base_score - lambda_redundancy * redundancy
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
                best_redundancy = redundancy
        assert best_idx is not None
        candidates[best_idx]["redundancy_penalty"] = float(best_redundancy)
        candidates[best_idx]["rerank_score"] = float(best_score)
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[idx] for idx in selected]


def _row_float(row: dict[str, object], key: str, default: float) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _grammar_variant_scores(rows: Sequence[dict[str, object]], score_variant: str) -> np.ndarray:
    token_nll = np.asarray([float(row.get("token_nll_p95", 0.0)) for row in rows], dtype=float)
    transition_nll = np.asarray([float(row.get("transition_nll_p95", 0.0)) for row in rows], dtype=float)
    rare_phrase = np.asarray([float(row.get("rare_phrase_fraction", 0.0)) for row in rows], dtype=float)
    longest_unseen = np.asarray([float(row.get("longest_unseen_phrase_len", 0.0)) for row in rows], dtype=float)
    if score_variant == "token_nll_p95":
        return minmax(token_nll)
    if score_variant == "grammar_phrase_mix":
        return minmax(0.70 * minmax(token_nll) + 0.30 * minmax(rare_phrase))
    return minmax(0.65 * minmax(token_nll) + 0.25 * minmax(transition_nll) + 0.10 * minmax(longest_unseen))


def _composite_split_score(row: dict[str, object], columns: Sequence[str]) -> float:
    values: list[float] = []
    for column in columns:
        try:
            value = float(row.get(column, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if np.isfinite(value):
            values.append(value)
    return float(sum(values) / len(values)) if values else 0.0


def _feature_subcluster_labels(
    rows: Sequence[dict[str, object]],
    vectors: np.ndarray,
    indices: Sequence[int],
    *,
    n_subclusters: int,
    max_subcluster_size: int,
    score_columns: Sequence[str],
    score_feature_weight: float,
    kmeans_iterations: int,
) -> list[int]:
    features = np.asarray(vectors[list(indices)], dtype=float)
    if score_feature_weight > 0.0:
        score_features = _score_feature_matrix(rows, indices, score_columns)
        if score_features.size:
            features = np.column_stack([features, score_feature_weight * score_features])
    labels = _deterministic_kmeans_labels(features, n_clusters=n_subclusters, iterations=kmeans_iterations)
    labels = _rebalance_oversized_labels(features, labels, max_size=max_subcluster_size)
    return _compact_labels_by_best_member(labels, rows, indices, score_columns)


def _score_feature_matrix(
    rows: Sequence[dict[str, object]],
    indices: Sequence[int],
    columns: Sequence[str],
) -> np.ndarray:
    if not columns:
        return np.empty((len(indices), 0), dtype=float)
    matrix = []
    for idx in indices:
        values = []
        for column in columns:
            try:
                value = float(rows[idx].get(column, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            values.append(value if np.isfinite(value) else 0.0)
        matrix.append(values)
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return np.empty((len(indices), 0), dtype=float)
    lo = np.min(values, axis=0)
    hi = np.max(values, axis=0)
    span = np.maximum(hi - lo, 1.0e-12)
    return (values - lo) / span


def _deterministic_kmeans_labels(features: np.ndarray, *, n_clusters: int, iterations: int) -> list[int]:
    values = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n_rows = len(values)
    if n_rows == 0:
        return []
    n_clusters = min(max(1, int(n_clusters)), n_rows)
    if n_clusters == n_rows:
        return list(range(n_rows))

    centers = _farthest_point_centers(values, n_clusters)
    labels = np.zeros(n_rows, dtype=int)
    for _iteration in range(iterations):
        distances = _squared_distances(values, centers)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels) and _iteration > 0:
            break
        labels = new_labels
        for cluster_id in range(n_clusters):
            members = values[labels == cluster_id]
            if len(members):
                centers[cluster_id] = np.mean(members, axis=0)
            else:
                farthest_idx = int(np.argmax(np.min(distances, axis=1)))
                centers[cluster_id] = values[farthest_idx]
                labels[farthest_idx] = cluster_id
    return labels.astype(int).tolist()


def _farthest_point_centers(values: np.ndarray, n_clusters: int) -> np.ndarray:
    mean = np.mean(values, axis=0)
    first = int(np.argmin(np.sum((values - mean) ** 2, axis=1)))
    center_indices = [first]
    min_distances = np.sum((values - values[first]) ** 2, axis=1)
    for _ in range(1, n_clusters):
        next_idx = int(np.argmax(min_distances))
        center_indices.append(next_idx)
        distances = np.sum((values - values[next_idx]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
    return values[center_indices].copy()


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def _compact_labels_by_best_member(
    labels: Sequence[int],
    rows: Sequence[dict[str, object]],
    indices: Sequence[int],
    score_columns: Sequence[str],
) -> list[int]:
    by_label: defaultdict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        by_label[int(label)].append(position)
    ordered_labels = sorted(
        by_label,
        key=lambda label: (
            -max(_composite_split_score(rows[indices[position]], score_columns) for position in by_label[label]),
            min(str(rows[indices[position]].get("sample_id", rows[indices[position]].get("worker_id", ""))) for position in by_label[label]),
        ),
    )
    mapping = {label: compact for compact, label in enumerate(ordered_labels)}
    return [int(mapping[int(label)]) for label in labels]


def _rebalance_oversized_labels(features: np.ndarray, labels: Sequence[int], *, max_size: int) -> list[int]:
    if max_size < 1:
        return [int(label) for label in labels]
    values = np.asarray(features, dtype=float)
    balanced = np.asarray(labels, dtype=int).copy()
    cluster_ids = sorted({int(label) for label in balanced})
    if not cluster_ids:
        return []

    while True:
        counts = {cluster_id: int(np.sum(balanced == cluster_id)) for cluster_id in cluster_ids}
        oversized = [cluster_id for cluster_id, count in counts.items() if count > max_size]
        underfull = [cluster_id for cluster_id, count in counts.items() if count < max_size]
        if not oversized or not underfull:
            break
        centers = {
            cluster_id: np.mean(values[balanced == cluster_id], axis=0)
            for cluster_id in cluster_ids
            if np.any(balanced == cluster_id)
        }
        moved = False
        for source in oversized:
            source_positions = np.flatnonzero(balanced == source)
            if len(source_positions) <= max_size:
                continue
            source_center = centers[source]
            farthest_first = source_positions[
                np.argsort(-np.sum((values[source_positions] - source_center) ** 2, axis=1))
            ]
            target_options = [target for target in underfull if target in centers and target != source]
            if not target_options:
                break
            for position in farthest_first:
                target = min(
                    target_options,
                    key=lambda candidate: float(np.sum((values[position] - centers[candidate]) ** 2)),
                )
                balanced[position] = target
                moved = True
                counts[target] += 1
                counts[source] -= 1
                if counts[target] >= max_size:
                    target_options = [candidate for candidate in target_options if candidate != target]
                    underfull = [candidate for candidate in underfull if candidate != target]
                if counts[source] <= max_size or not target_options:
                    break
        if not moved:
            break
    return balanced.astype(int).tolist()


def _refresh_split_medoid_distances(rows: list[dict[str, object]], vectors: np.ndarray) -> None:
    by_cluster: defaultdict[int, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_cluster[int(row["new_cluster_id"])].append(idx)
    for indices in by_cluster.values():
        cluster_vectors = vectors[indices]
        if len(indices) == 1:
            medoid_idx = indices[0]
        else:
            sims = cluster_vectors @ cluster_vectors.T
            medoid_idx = indices[int(np.argmax(np.mean(sims, axis=1)))]
        medoid = vectors[medoid_idx]
        for idx in indices:
            rows[idx]["distance_to_new_cluster_medoid"] = float(1.0 - vectors[idx] @ medoid)


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
