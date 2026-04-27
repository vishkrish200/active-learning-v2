from __future__ import annotations

import numpy as np

from marginal_value.indexing.knn_features import normalize_rows


def new_batch_support_features(embeddings: np.ndarray, *, n_clusters: int | None = None) -> list[dict[str, float]]:
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    if len(vectors) == 0:
        return []

    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, np.nan)
    density = np.nanmean(np.maximum(similarities, 0.0), axis=1)
    density = np.nan_to_num(density, nan=0.0)

    if n_clusters is None:
        n_clusters = max(1, min(32, int(round(np.sqrt(len(vectors))))))
    labels, centers = kmeans(vectors, n_clusters=n_clusters)
    sizes = np.bincount(labels, minlength=n_clusters)

    rows: list[dict[str, float]] = []
    for idx, label in enumerate(labels):
        center = centers[label]
        distance_to_medoid = float(1.0 - vectors[idx] @ center / max(np.linalg.norm(center), 1.0e-12))
        rows.append(
            {
                "new_batch_density": float(density[idx]),
                "new_cluster_id": int(label),
                "new_cluster_size": int(sizes[label]),
                "is_singleton": bool(sizes[label] == 1),
                "distance_to_new_cluster_medoid": distance_to_medoid,
            }
        )
    return rows


def kmeans(
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    n_iter: int = 30,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(embeddings, dtype=float)
    if len(data) == 0:
        raise ValueError("cannot cluster empty embeddings")
    k = max(1, min(n_clusters, len(data)))
    rng = np.random.default_rng(seed)
    centers = data[rng.choice(len(data), size=k, replace=False)].copy()

    labels = np.zeros(len(data), dtype=int)
    for _ in range(n_iter):
        distances = _squared_distances(data, centers)
        labels = np.argmin(distances, axis=1)
        for idx in range(k):
            members = data[labels == idx]
            if len(members):
                centers[idx] = np.mean(members, axis=0)
    return labels, centers


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)

