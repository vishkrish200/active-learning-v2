from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from marginal_value.indexing.cosine_search import cosine_knn as search_cosine_knn
from marginal_value.indexing.knn_features import normalize_rows


class EvalLeakageError(RuntimeError):
    """Raised when an eval config would leak held-out data into support/training."""


def load_eval_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_eval_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    checkpoint = _required_mapping(config, "checkpoint")
    splits = _required_mapping(config, "splits")
    eval_config = _required_mapping(config, "eval")
    acceptance = config.get("acceptance", {})
    if acceptance is not None and not isinstance(acceptance, dict):
        raise ValueError("Eval config acceptance section must be an object when provided.")

    if execution.get("provider") != "modal":
        raise ValueError("Encoder evaluation must run on Modal.")
    if execution.get("gpu") != "H100":
        raise ValueError("Encoder evaluation must request an H100.")
    if not str(data.get("root", "")).startswith("/data"):
        raise ValueError("Eval data root must be mounted under /data.")
    if not str(artifacts.get("root", "")).startswith("/artifacts"):
        raise ValueError("Eval artifacts root must be mounted under /artifacts.")
    if "pretrain_only" not in str(checkpoint.get("path", "")):
        raise EvalLeakageError("Evaluation checkpoint must be the pretrain-only checkpoint.")

    support_split = splits.get("support_split")
    query_split = splits.get("query_split")
    if support_split == query_split:
        raise EvalLeakageError("support_split and query_split must be different.")
    if support_split != "pretrain" or query_split != "val":
        raise EvalLeakageError("Current evaluation must use pretrain support and val query.")

    if int(data.get("feature_dim", 0)) <= 0:
        raise ValueError("data.feature_dim must be positive.")
    k_values = eval_config.get("k_values", [])
    if not k_values or any(int(k) <= 0 for k in k_values):
        raise ValueError("eval.k_values must contain positive integers.")
    if acceptance:
        if float(acceptance.get("min_effective_rank", 0.0)) < 0.0:
            raise ValueError("acceptance.min_effective_rank must be non-negative.")
        if float(acceptance.get("min_mean_pairwise_cosine_distance", 0.0)) < 0.0:
            raise ValueError("acceptance.min_mean_pairwise_cosine_distance must be non-negative.")


def cosine_knn(support: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return search_cosine_knn(support, query, k=k, backend="auto")


def evaluate_retrieval(
    *,
    support_encoder: np.ndarray,
    query_encoder: np.ndarray,
    support_baseline: np.ndarray,
    query_baseline: np.ndarray,
    k_values: Iterable[int],
) -> dict[str, Any]:
    k_list = [int(k) for k in k_values]
    return {
        "n_support": int(len(support_encoder)),
        "n_query": int(len(query_encoder)),
        "embedding_dim": int(np.asarray(support_encoder).shape[1]),
        "baseline_dim": int(np.asarray(support_baseline).shape[1]),
        "encoder": _retrieval_summary(support_encoder, query_encoder, k_list)
        | {"diagnostics": embedding_diagnostics(np.vstack([support_encoder, query_encoder]))},
        "baseline": _retrieval_summary(support_baseline, query_baseline, k_list)
        | {"diagnostics": embedding_diagnostics(np.vstack([support_baseline, query_baseline]))},
    }


def embedding_diagnostics(embeddings: np.ndarray) -> dict[str, float]:
    values = np.asarray(embeddings, dtype=float)
    if values.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    centered = values - np.mean(values, axis=0, keepdims=True)
    dim_std = np.std(values, axis=0)
    if min(values.shape) <= 1 or not np.any(centered):
        effective_rank = 0.0
    else:
        singular_values = np.linalg.svd(centered, compute_uv=False)
        energy = singular_values**2
        probs = energy / max(float(np.sum(energy)), 1.0e-12)
        entropy = -float(np.sum(probs * np.log(probs + 1.0e-12)))
        effective_rank = float(np.exp(entropy))

    sample = values[: min(len(values), 512)]
    if len(sample) < 2:
        mean_pairwise_cosine_distance = 0.0
    else:
        normalized = normalize_rows(sample)
        sims = normalized @ normalized.T
        upper = sims[np.triu_indices_from(sims, k=1)]
        mean_pairwise_cosine_distance = float(np.mean(1.0 - upper))

    return {
        "mean_dimension_std": float(np.mean(dim_std)),
        "min_dimension_std": float(np.min(dim_std)) if len(dim_std) else 0.0,
        "max_dimension_std": float(np.max(dim_std)) if len(dim_std) else 0.0,
        "effective_rank": effective_rank,
        "mean_pairwise_cosine_distance": mean_pairwise_cosine_distance,
    }


def write_eval_report(report: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _retrieval_summary(support: np.ndarray, query: np.ndarray, k_values: list[int]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for k in k_values:
        distances, _indices = cosine_knn(support, query, k)
        summary[f"mean_knn_d{k}"] = float(np.mean(distances[:, : min(k, distances.shape[1])]))
        summary[f"p95_knn_d{k}"] = float(np.percentile(distances[:, : min(k, distances.shape[1])], 95))
    return summary


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Eval config must include a '{key}' object.")
    return value
