from __future__ import annotations

from typing import Iterable

import numpy as np


def rank_by_score(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_values = np.asarray(scores, dtype=float)
    label_values = np.asarray(labels, dtype=int)
    order = np.argsort(-score_values, kind="mergesort")
    return score_values[order], label_values[order], order


def precision_at_k(labels: Iterable[int], k: int) -> float:
    values = np.asarray(list(labels), dtype=int)[:k]
    return float(np.mean(values > 0)) if len(values) else 0.0


def average_precision_at_k(labels: Iterable[int], k: int) -> float:
    values = np.asarray(list(labels), dtype=int)[:k]
    if len(values) == 0:
        return 0.0
    precisions = []
    positives = 0
    for idx, label in enumerate(values, start=1):
        if label > 0:
            positives += 1
            precisions.append(positives / idx)
    total_positive = int(np.sum(np.asarray(list(labels), dtype=int) > 0))
    if total_positive == 0:
        return 0.0
    return float(np.sum(precisions) / min(total_positive, k))


def ndcg_at_k(labels: Iterable[int], k: int) -> float:
    values = np.asarray(list(labels), dtype=float)[:k]
    if len(values) == 0:
        return 0.0
    gains = (2.0**values) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(values) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(np.asarray(list(labels), dtype=float))[::-1][:k]
    ideal_gains = (2.0**ideal) - 1.0
    idcg = float(np.sum(ideal_gains * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def cluster_diversity_at_k(labels: Iterable[int], clusters: Iterable[int], k: int) -> float:
    label_values = np.asarray(list(labels), dtype=int)[:k]
    cluster_values = np.asarray(list(clusters), dtype=int)[:k]
    positive_clusters = {int(cluster) for label, cluster in zip(label_values, cluster_values) if label > 0}
    denom = max(1, int(np.sum(label_values > 0)))
    return float(len(positive_clusters) / denom)


def summarize_ranked_scores(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray | None = None,
    corruptions: np.ndarray | None = None,
    k_values: Iterable[int] = (10, 50, 100, 200),
) -> dict[str, float]:
    ranked_scores, ranked_labels, order = rank_by_score(scores, labels)
    ranked_clusters = clusters[order] if clusters is not None else np.arange(len(order))
    ranked_corruptions = corruptions[order] if corruptions is not None else np.zeros(len(order), dtype=bool)
    summary: dict[str, float] = {
        "score_mean": float(np.mean(ranked_scores)) if len(ranked_scores) else 0.0,
        "positive_rate": float(np.mean(np.asarray(labels, dtype=int) > 0)) if len(labels) else 0.0,
        "corruption_rate": float(np.mean(np.asarray(corruptions, dtype=bool))) if corruptions is not None and len(corruptions) else 0.0,
    }
    for k in k_values:
        k_eff = min(int(k), len(ranked_labels))
        summary[f"precision@{k}"] = precision_at_k(ranked_labels, k_eff)
        summary[f"ap@{k}"] = average_precision_at_k(ranked_labels, k_eff)
        summary[f"ndcg@{k}"] = ndcg_at_k(ranked_labels, k_eff)
        summary[f"cluster_diversity@{k}"] = cluster_diversity_at_k(ranked_labels, ranked_clusters, k_eff)
        summary[f"corruption_rate@{k}"] = float(np.mean(ranked_corruptions[:k_eff])) if k_eff else 0.0
    return summary
