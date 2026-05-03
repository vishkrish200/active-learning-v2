from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from marginal_value.indexing.cosine_search import cosine_knn
from marginal_value.indexing.knn_features import normalize_rows


BASELINE_POLICY_NAMES = (
    "random_valid",
    "quality_only",
    "old_novelty_only",
    "kcenter_greedy_quality_gated",
    "window_shape_stats_q85_stat90_abs60_clustercap2",
)


def random_valid_order(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    quality_threshold: float = 0.45,
) -> list[int]:
    valid = [idx for idx, row in enumerate(rows) if _safe_float(row.get("quality_score", 0.0)) >= quality_threshold]
    invalid = [idx for idx in range(len(rows)) if idx not in set(valid)]
    rng = np.random.default_rng(int(seed))
    valid_order = list(rng.permutation(valid).astype(int)) if valid else []
    invalid_order = list(rng.permutation(invalid).astype(int)) if invalid else []
    return [*valid_order, *invalid_order]


def quality_only_order(rows: Sequence[Mapping[str, object]]) -> list[int]:
    return [
        int(idx)
        for idx in sorted(
            range(len(rows)),
            key=lambda idx: (-_safe_float(rows[idx].get("quality_score", 0.0)), _sample_key(rows[idx], idx)),
        )
    ]


def old_novelty_only_order(
    support_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    *,
    k: int = 10,
) -> tuple[list[int], np.ndarray]:
    distances, _indices = cosine_knn(support_embeddings, candidate_embeddings, k=k, backend="auto")
    novelty = np.mean(distances, axis=1)
    order = [
        int(idx)
        for idx in sorted(range(len(novelty)), key=lambda idx: (-float(novelty[idx]), idx))
    ]
    return order, novelty


def blended_old_novelty_order(
    novelty_left: np.ndarray,
    novelty_right: np.ndarray,
    *,
    alpha: float,
) -> list[int]:
    left = np.asarray(novelty_left, dtype=float)
    right = np.asarray(novelty_right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Blended novelty arrays must have the same shape.")
    weight = float(alpha)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("alpha must be in [0, 1].")
    scores = weight * _minmax(left) + (1.0 - weight) * _minmax(right)
    return [
        int(idx)
        for idx in sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    ]


def kcenter_greedy_quality_gated_order(
    support_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    *,
    quality_threshold: float = 0.45,
    max_stationary_fraction: float | None = None,
    max_abs_value: float | None = None,
) -> list[int]:
    if len(rows) != len(candidate_embeddings):
        raise ValueError("rows and candidate_embeddings must have the same length.")
    eligible = [
        idx
        for idx, row in enumerate(rows)
        if _passes_quality_gates(
            row,
            quality_threshold=quality_threshold,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
    ]
    selected = _farthest_first_order(
        support_embeddings=support_embeddings,
        candidate_embeddings=candidate_embeddings,
        rows=rows,
        eligible_indices=eligible,
    )
    selected_set = set(selected)
    fallback = [
        idx
        for idx in quality_only_order(rows)
        if idx not in selected_set
    ]
    return [*selected, *fallback]


def blended_kcenter_greedy_quality_gated_order(
    support_embeddings_left: np.ndarray,
    candidate_embeddings_left: np.ndarray,
    support_embeddings_right: np.ndarray,
    candidate_embeddings_right: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    *,
    alpha: float,
    quality_threshold: float = 0.45,
    max_stationary_fraction: float | None = None,
    max_abs_value: float | None = None,
) -> list[int]:
    if len(rows) != len(candidate_embeddings_left) or len(rows) != len(candidate_embeddings_right):
        raise ValueError("rows and candidate embeddings must have the same length.")
    weight = float(alpha)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("alpha must be in [0, 1].")
    eligible = [
        idx
        for idx, row in enumerate(rows)
        if _passes_quality_gates(
            row,
            quality_threshold=quality_threshold,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
    ]
    selected = _blended_farthest_first_order(
        support_embeddings_left=support_embeddings_left,
        candidate_embeddings_left=candidate_embeddings_left,
        support_embeddings_right=support_embeddings_right,
        candidate_embeddings_right=candidate_embeddings_right,
        rows=rows,
        eligible_indices=eligible,
        alpha=weight,
    )
    selected_set = set(selected)
    fallback = [
        idx
        for idx in quality_only_order(rows)
        if idx not in selected_set
    ]
    return [*selected, *fallback]


def trace_hygiene_rerank_order(
    base_order: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    *,
    spike_rate_threshold: float = 0.025,
) -> list[int]:
    """Move trace-failed rows behind trace-pass rows while preserving base order."""

    return _trace_rerank_order(
        base_order,
        rows,
        failed=lambda row: _trace_hygiene_failed(row, spike_rate_threshold=spike_rate_threshold),
    )


def trace_artifact_rerank_order(
    base_order: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> list[int]:
    """Move likely trace artifacts behind trace-pass rows while preserving base order."""

    return _trace_rerank_order(
        base_order,
        rows,
        failed=_trace_artifact_failed,
    )


def _trace_rerank_order(
    base_order: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    *,
    failed,
) -> list[int]:
    ordered = [int(idx) for idx in base_order]
    if len(set(ordered)) != len(ordered):
        raise ValueError("base_order must not contain duplicate indices.")
    missing = [idx for idx in range(len(rows)) if idx not in set(ordered)]
    full_order = [*ordered, *missing]
    rank_by_idx = {int(idx): order_idx for order_idx, idx in enumerate(full_order)}
    return [
        int(idx)
        for idx in sorted(
            full_order,
            key=lambda idx: (
                failed(rows[int(idx)]),
                rank_by_idx[int(idx)],
            ),
        )
    ]


def learned_score_quality_gated_kcenter_order(
    support_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    *,
    quality_threshold: float = 0.85,
    max_stationary_fraction: float | None = 0.90,
    max_abs_value: float | None = 60.0,
    max_selected: int | None = None,
    pool_multiplier: float = 1.5,
    score_key: str = "learned_score",
) -> list[int]:
    if len(rows) != len(candidate_embeddings):
        raise ValueError("rows and candidate_embeddings must have the same length.")
    selected_limit = len(rows) if max_selected is None else min(len(rows), max(0, int(max_selected)))
    eligible = [
        idx
        for idx, row in enumerate(rows)
        if _passes_quality_gates(
            row,
            quality_threshold=quality_threshold,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
    ]
    learned_ranked = sorted(
        eligible,
        key=lambda idx: (
            -_safe_float(rows[int(idx)].get(score_key, 0.0)),
            -_safe_float(rows[int(idx)].get("quality_score", 0.0)),
            _sample_key(rows[int(idx)], int(idx)),
        ),
    )
    if selected_limit <= 0:
        selected: list[int] = []
    else:
        pool_size = min(
            len(learned_ranked),
            max(selected_limit, int(np.ceil(selected_limit * max(1.0, float(pool_multiplier))))),
        )
        selected = _farthest_first_order(
            support_embeddings=support_embeddings,
            candidate_embeddings=candidate_embeddings,
            rows=rows,
            eligible_indices=learned_ranked[:pool_size],
            tie_score_key=score_key,
        )
    selected_set = set(selected)
    eligible_fallback = [idx for idx in learned_ranked if idx not in selected_set]
    invalid_fallback = [
        idx
        for idx in sorted(
            set(range(len(rows))) - set(eligible),
            key=lambda idx: (
                -_safe_float(rows[int(idx)].get(score_key, 0.0)),
                -_safe_float(rows[int(idx)].get("quality_score", 0.0)),
                _sample_key(rows[int(idx)], int(idx)),
            ),
        )
    ]
    return [*selected, *eligible_fallback, *invalid_fallback]


def window_shape_deterministic_baseline_order(
    rows: Sequence[Mapping[str, object]],
    *,
    quality_threshold: float = 0.85,
    max_stationary_fraction: float = 0.90,
    max_abs_value: float = 60.0,
    cluster_cap: int = 2,
    cluster_key: str = "new_cluster_id",
) -> list[int]:
    ranked = sorted(
        range(len(rows)),
        key=lambda idx: (
            not _passes_quality_gates(
                rows[idx],
                quality_threshold=quality_threshold,
                max_stationary_fraction=max_stationary_fraction,
                max_abs_value=max_abs_value,
            ),
            -_safe_float(rows[idx].get("old_novelty_score", rows[idx].get("old_knn_distance", 0.0))),
            -_safe_float(rows[idx].get("quality_score", 0.0)),
            _sample_key(rows[idx], idx),
        ),
    )
    if cluster_cap <= 0:
        return [int(idx) for idx in ranked]
    selected: list[int] = []
    overflow: list[int] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for idx in ranked:
        cluster = str(rows[idx].get(cluster_key, rows[idx].get("source_group_id", idx)))
        if counts[cluster] < cluster_cap:
            selected.append(int(idx))
            counts[cluster] += 1
        else:
            overflow.append(int(idx))
    return [*selected, *overflow]


def _farthest_first_order(
    *,
    support_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    eligible_indices: Sequence[int],
    tie_score_key: str | None = None,
) -> list[int]:
    if not eligible_indices:
        return []
    candidates = normalize_rows(np.asarray(candidate_embeddings, dtype=float))
    support = normalize_rows(np.asarray(support_embeddings, dtype=float))
    if support.ndim != 2 or candidates.ndim != 2 or support.shape[1] != candidates.shape[1]:
        raise ValueError("support and candidate embeddings must be 2D arrays with the same dimension.")
    nearest_similarity = np.max(candidates @ support.T, axis=1) if len(support) else np.full(len(candidates), -1.0)
    remaining = set(int(idx) for idx in eligible_indices)
    ordered: list[int] = []
    while remaining:
        best = sorted(
            remaining,
            key=lambda idx: (
                -(1.0 - float(nearest_similarity[int(idx)])),
                -_safe_float(rows[int(idx)].get(tie_score_key, 0.0)) if tie_score_key else 0.0,
                -_safe_float(rows[int(idx)].get("quality_score", 0.0)),
                _sample_key(rows[int(idx)], int(idx)),
            ),
        )[0]
        ordered.append(best)
        remaining.remove(best)
        nearest_similarity = np.maximum(nearest_similarity, candidates @ candidates[best])
    return ordered


def _blended_farthest_first_order(
    *,
    support_embeddings_left: np.ndarray,
    candidate_embeddings_left: np.ndarray,
    support_embeddings_right: np.ndarray,
    candidate_embeddings_right: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    eligible_indices: Sequence[int],
    alpha: float,
) -> list[int]:
    if not eligible_indices:
        return []
    candidates_left = normalize_rows(np.asarray(candidate_embeddings_left, dtype=float))
    support_left = normalize_rows(np.asarray(support_embeddings_left, dtype=float))
    candidates_right = normalize_rows(np.asarray(candidate_embeddings_right, dtype=float))
    support_right = normalize_rows(np.asarray(support_embeddings_right, dtype=float))
    _validate_search_shapes(support_left, candidates_left, side="left")
    _validate_search_shapes(support_right, candidates_right, side="right")
    if len(candidates_left) != len(candidates_right):
        raise ValueError("Left and right candidate embeddings must have the same row count.")

    nearest_left = np.max(candidates_left @ support_left.T, axis=1) if len(support_left) else np.full(len(candidates_left), -1.0)
    nearest_right = np.max(candidates_right @ support_right.T, axis=1) if len(support_right) else np.full(len(candidates_right), -1.0)
    remaining = set(int(idx) for idx in eligible_indices)
    ordered: list[int] = []
    while remaining:
        remaining_indices = np.asarray(sorted(remaining), dtype=int)
        distance_left = 1.0 - nearest_left[remaining_indices]
        distance_right = 1.0 - nearest_right[remaining_indices]
        scores = alpha * _minmax(distance_left) + (1.0 - alpha) * _minmax(distance_right)
        score_by_idx = {int(idx): float(score) for idx, score in zip(remaining_indices, scores, strict=True)}
        best = sorted(
            remaining,
            key=lambda idx: (
                -score_by_idx[int(idx)],
                -_safe_float(rows[int(idx)].get("quality_score", 0.0)),
                _sample_key(rows[int(idx)], int(idx)),
            ),
        )[0]
        ordered.append(best)
        remaining.remove(best)
        nearest_left = np.maximum(nearest_left, candidates_left @ candidates_left[best])
        nearest_right = np.maximum(nearest_right, candidates_right @ candidates_right[best])
    return ordered


def _validate_search_shapes(support: np.ndarray, candidates: np.ndarray, *, side: str) -> None:
    if support.ndim != 2 or candidates.ndim != 2 or support.shape[1] != candidates.shape[1]:
        raise ValueError(f"{side} support and candidate embeddings must be 2D arrays with the same dimension.")


def _passes_quality_gates(
    row: Mapping[str, object],
    *,
    quality_threshold: float,
    max_stationary_fraction: float | None,
    max_abs_value: float | None,
) -> bool:
    if _safe_float(row.get("quality_score", 0.0)) < float(quality_threshold):
        return False
    if max_stationary_fraction is not None and _safe_float(row.get("stationary_fraction", 0.0)) > max_stationary_fraction:
        return False
    if max_abs_value is not None and _safe_float(row.get("max_abs_value", 0.0)) > max_abs_value:
        return False
    return True


def _trace_hygiene_failed(row: Mapping[str, object], *, spike_rate_threshold: float) -> bool:
    if _safe_float(row.get("quality__spike_rate", 0.0)) > float(spike_rate_threshold):
        return True
    verdict = str(row.get("trace__verdict", "")).strip().lower()
    return verdict in {"likely_artifact", "mostly_stationary"}


def _trace_artifact_failed(row: Mapping[str, object]) -> bool:
    verdict = str(row.get("trace__verdict", "")).strip().lower()
    return verdict == "likely_artifact"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _sample_key(row: Mapping[str, object], idx: int) -> str:
    return str(row.get("sample_id", row.get("worker_id", f"{idx:012d}")))


def _minmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    lo = float(np.min(array))
    hi = float(np.max(array))
    if hi - lo < 1.0e-12:
        return np.zeros_like(array, dtype=float)
    return (array - lo) / (hi - lo)
