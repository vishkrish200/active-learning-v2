from __future__ import annotations

from typing import Literal

import numpy as np

from marginal_value.indexing.knn_features import normalize_rows


BackendName = Literal["auto", "numpy", "torch"]


def cosine_knn(
    support: np.ndarray,
    query: np.ndarray,
    k: int,
    *,
    backend: BackendName = "auto",
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine-distance nearest neighbors with optional CUDA acceleration."""

    support_vectors = np.asarray(support, dtype=float)
    query_vectors = np.asarray(query, dtype=float)
    if support_vectors.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError("support and query embeddings must be 2D arrays")
    if support_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("support and query embeddings must have the same dimension")
    k_eff = min(int(k), len(support_vectors))
    if k_eff <= 0:
        raise ValueError("support must contain at least one embedding")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    if _should_use_torch(backend):
        return _torch_cosine_knn(support_vectors, query_vectors, k_eff, batch_size=int(batch_size))
    if backend == "torch":
        raise RuntimeError("Torch/CUDA cosine search requested, but torch.cuda is not available.")
    return _numpy_cosine_knn(support_vectors, query_vectors, k_eff)


def mean_nearest_cosine_distance(
    query: np.ndarray,
    support: np.ndarray,
    *,
    backend: BackendName = "auto",
    batch_size: int = 4096,
) -> float:
    distances, _indices = cosine_knn(support, query, k=1, backend=backend, batch_size=batch_size)
    return float(np.mean(np.maximum(distances[:, 0], 0.0)))


def _numpy_cosine_knn(support: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    support_vectors = normalize_rows(support)
    query_vectors = normalize_rows(query)
    similarities = query_vectors @ support_vectors.T
    order = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    indices = np.empty_like(order)
    distances = np.empty((len(query_vectors), k), dtype=float)
    for row_idx in range(len(query_vectors)):
        row_order = order[row_idx]
        row_sims = similarities[row_idx, row_order]
        sorted_local = np.lexsort((row_order, -row_sims))
        sorted_indices = row_order[sorted_local]
        indices[row_idx] = sorted_indices
        distances[row_idx] = 1.0 - similarities[row_idx, sorted_indices]
    return distances, indices


def _torch_cosine_knn(support: np.ndarray, query: np.ndarray, k: int, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    support_tensor = torch.as_tensor(support, dtype=torch.float32, device=device)
    support_tensor = F.normalize(support_tensor, dim=1, eps=1.0e-12)
    distance_batches: list[np.ndarray] = []
    index_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query), batch_size):
            batch = torch.as_tensor(query[start : start + batch_size], dtype=torch.float32, device=device)
            batch = F.normalize(batch, dim=1, eps=1.0e-12)
            similarities = batch @ support_tensor.T
            values, indices = torch.topk(similarities, k=k, dim=1, largest=True, sorted=True)
            distance_batches.append((1.0 - values).detach().cpu().numpy())
            index_batches.append(indices.detach().cpu().numpy())
    return np.vstack(distance_batches), np.vstack(index_batches)


def _should_use_torch(backend: BackendName) -> bool:
    if backend == "numpy":
        return False
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())
