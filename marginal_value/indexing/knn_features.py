from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1.0e-12)


@dataclass
class ExactKNNIndex:
    embeddings: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "ExactKNNIndex":
        self.embeddings = normalize_rows(np.asarray(embeddings, dtype=float))
        return self

    def query(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self.embeddings is None:
            raise RuntimeError("fit must be called before query")
        query_vectors = normalize_rows(np.asarray(queries, dtype=float))
        similarities = query_vectors @ self.embeddings.T
        k_eff = min(k, self.embeddings.shape[0])
        order = np.argpartition(-similarities, kth=k_eff - 1, axis=1)[:, :k_eff]
        row_indices = np.arange(similarities.shape[0])[:, None]
        sorted_local = np.argsort(-similarities[row_indices, order], axis=1)
        indices = order[row_indices, sorted_local]
        distances = 1.0 - similarities[row_indices, indices]
        return distances, indices


def build_old_support_features(
    index: ExactKNNIndex,
    query_embeddings: np.ndarray,
    *,
    ks: Iterable[int] = (1, 5, 10, 50, 100),
) -> list[dict[str, float]]:
    max_k = max(ks)
    distances, _indices = index.query(query_embeddings, k=max_k)
    rows: list[dict[str, float]] = []
    for row_distances in distances:
        features: dict[str, float] = {}
        for k in ks:
            k_eff = min(k, len(row_distances))
            value = float(np.mean(row_distances[:k_eff]))
            if k == 1:
                features["old_knn_d1"] = value
            features[f"old_knn_d{k}_mean"] = value
            features[f"old_knn_d{k}_std"] = float(np.std(row_distances[:k_eff]))
        features.setdefault("old_knn_d1", float(row_distances[0]))
        rows.append(features)
    return rows

