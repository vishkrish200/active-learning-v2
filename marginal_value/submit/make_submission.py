from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from marginal_value.indexing.knn_features import normalize_rows


def diversity_rerank(
    rows: Iterable[dict[str, object]],
    embeddings: np.ndarray,
    *,
    lambda_redundancy: float = 0.25,
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
        for idx in remaining:
            base_score = float(candidates[idx].get("final_score", 0.0))
            redundancy = 0.0
            if selected:
                redundancy = float(np.max(vectors[idx] @ vectors[selected].T))
            mmr_score = base_score - lambda_redundancy * redundancy
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
        assert best_idx is not None
        candidates[best_idx]["redundancy_penalty"] = float(
            0.0 if not selected else np.max(vectors[best_idx] @ vectors[selected].T)
        )
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[idx] for idx in selected]


def build_submission_rows(scored_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = sorted((dict(row) for row in scored_rows), key=lambda item: float(item["final_score"]), reverse=True)
    submission: list[dict[str, object]] = []
    for rank, row in enumerate(rows, start=1):
        submission.append(
            {
                "worker_id": row["worker_id"],
                "rank": rank,
                "score": float(row["final_score"]),
                "quality_score": float(row.get("quality_score", 1.0)),
                "reason_code": row.get("reason_code", ""),
            }
        )
    return submission


def write_submission(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(build_submission_rows(rows)).to_csv(output, index=False)


def write_diagnostics(rows: Iterable[dict[str, object]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output, index=False)

