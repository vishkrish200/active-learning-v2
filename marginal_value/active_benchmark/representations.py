from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from marginal_value.active_benchmark.schema import BenchmarkClip


def clip_lookup(clips: Sequence[BenchmarkClip]) -> dict[str, BenchmarkClip]:
    lookup = {clip.sample_id: clip for clip in clips}
    if len(lookup) != len(clips):
        raise ValueError("Benchmark clips must have unique sample_id values.")
    return lookup


def stack_embeddings(
    clips_by_id: Mapping[str, BenchmarkClip],
    sample_ids: Sequence[str],
    *,
    representation: str,
) -> np.ndarray:
    rows = []
    for sample_id in sample_ids:
        clip = clips_by_id[str(sample_id)]
        if representation not in clip.embeddings:
            raise ValueError(f"Clip {sample_id} is missing representation '{representation}'.")
        rows.append(np.asarray(clip.embeddings[representation], dtype=float).reshape(-1))
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = rows[0].shape[0]
    if any(row.shape[0] != width for row in rows):
        raise ValueError(f"Representation '{representation}' has inconsistent dimensions.")
    return np.vstack(rows)
