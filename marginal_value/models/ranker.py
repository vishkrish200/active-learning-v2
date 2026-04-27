from __future__ import annotations

from typing import Iterable

import numpy as np


def score_candidates(feature_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    rows = [dict(row) for row in feature_rows]
    novelty_values = np.array([_old_novelty(row) for row in rows], dtype=float)
    novelty_norm = 0.50 * _minmax(novelty_values) + 0.50 * np.clip(novelty_values, 0.0, 1.0)

    for row, novelty in zip(rows, novelty_norm):
        quality = float(row.get("quality_score", 1.0))
        support = _support_score(row)
        grammar = _grammar_score(row)
        singleton_penalty = 0.75 if bool(row.get("is_singleton", False)) else 1.0

        ranker_score = 0.50 * novelty + 0.30 * support + 0.20 * grammar
        final_score = quality * singleton_penalty * ranker_score
        row["old_novelty_score"] = float(novelty)
        row["new_support_score"] = float(support)
        row["grammar_score"] = float(grammar)
        row["final_score"] = float(np.clip(final_score, 0.0, 1.0))
        row["reason_code"] = assign_reason_code(row)
        scored.append(row)

    scored.sort(key=lambda item: float(item["final_score"]), reverse=True)
    return scored


def assign_reason_code(row: dict[str, object]) -> str:
    quality = float(row.get("quality_score", 1.0))
    if quality < 0.45:
        return "LOW_QUALITY"

    novelty = float(row.get("old_novelty_score", _old_novelty(row)))
    support = _support_score(row)
    rare_phrase = float(row.get("rare_phrase_fraction", 0.0))
    token_nll = float(row.get("token_nll_p95", row.get("token_nll_mean", 0.0)))

    if novelty >= 0.50 and support >= 0.45:
        return "COHESIVE_NEW_WORKFLOW"
    if rare_phrase >= 0.20 or token_nll >= 3.0:
        return "RARE_TEMPORAL_COMPOSITION"
    if novelty >= 0.65:
        return "HIGH_NOVELTY_SINGLETON"
    if support >= 0.65 and novelty < 0.30:
        return "REDUNDANT_KNOWN_WORKFLOW"
    return "RARE_MOTION_PRIMITIVES" if grammar_signal(row) > 0.25 else "REDUNDANT_KNOWN_WORKFLOW"


def grammar_signal(row: dict[str, object]) -> float:
    return _grammar_score(row)


def _old_novelty(row: dict[str, object]) -> float:
    for key in ("old_knn_d50_mean", "old_knn_d10_mean", "old_knn_d5_mean", "old_knn_d3_mean", "old_knn_d1"):
        if key in row:
            return float(row[key])
    return 0.0


def _support_score(row: dict[str, object]) -> float:
    density = float(row.get("new_batch_density", 0.0))
    cluster_size = float(row.get("new_cluster_size", 1.0))
    cluster_bonus = np.log1p(max(cluster_size, 1.0)) / np.log(11.0)
    return float(np.clip(0.65 * density + 0.35 * cluster_bonus, 0.0, 1.0))


def _grammar_score(row: dict[str, object]) -> float:
    nll = float(row.get("token_nll_p95", row.get("token_nll_mean", 0.0)))
    rare = float(row.get("rare_phrase_fraction", 0.0))
    nll_score = 1.0 - np.exp(-max(nll, 0.0) / 4.0)
    return float(np.clip(0.60 * nll_score + 0.40 * rare, 0.0, 1.0))


def _minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    span = float(np.max(values) - np.min(values))
    if span < 1.0e-12:
        return np.clip(values, 0.0, 1.0)
    return (values - float(np.min(values))) / span
