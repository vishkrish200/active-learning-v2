from __future__ import annotations

import numpy as np


def ndcg_at_k(relevances: list[float] | np.ndarray, k: int) -> float:
    values = np.asarray(relevances, dtype=float)[:k]
    if len(values) == 0:
        return 0.0
    gains = (2.0**values) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(values) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(np.asarray(relevances, dtype=float))[::-1][:k]
    ideal_gains = (2.0**ideal) - 1.0
    idcg = float(np.sum(ideal_gains * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(labels: list[int] | np.ndarray, k: int) -> float:
    values = np.asarray(labels, dtype=int)[:k]
    return float(np.mean(values > 0)) if len(values) else 0.0


def corruption_rate_at_k(is_corrupt: list[bool] | np.ndarray, k: int) -> float:
    values = np.asarray(is_corrupt, dtype=bool)[:k]
    return float(np.mean(values)) if len(values) else 0.0

