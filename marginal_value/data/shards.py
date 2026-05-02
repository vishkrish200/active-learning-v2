from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SHARD_FORMAT_VERSION = "full_support_shards_v1"


@dataclass(frozen=True)
class ShardWrite:
    index: int
    path: Path
    relpath: str
    n_clips: int
    sample_ids: list[str]


def write_clip_shard(
    path: Path,
    *,
    sample_ids: Sequence[str],
    worker_ids: Sequence[str],
    source_group_ids: Sequence[str],
    splits: Sequence[str],
    urls: Sequence[str],
    quality: Mapping[str, np.ndarray],
    representations: Mapping[str, np.ndarray],
    imu_samples: np.ndarray | None = None,
    imu_lengths: np.ndarray | None = None,
    compress: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format_version": np.asarray(SHARD_FORMAT_VERSION),
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "worker_ids": np.asarray(worker_ids, dtype=str),
        "source_group_ids": np.asarray(source_group_ids, dtype=str),
        "splits": np.asarray(splits, dtype=str),
        "urls": np.asarray(urls, dtype=str),
    }
    for key, value in quality.items():
        payload[key] = np.asarray(value, dtype="float32")
    for representation, matrix in representations.items():
        payload[f"rep__{representation}"] = np.asarray(matrix, dtype="float32")
    if imu_samples is not None:
        payload["imu_samples"] = np.asarray(imu_samples, dtype="float32")
    if imu_lengths is not None:
        payload["imu_lengths"] = np.asarray(imu_lengths, dtype="int32")
    writer = np.savez_compressed if compress else np.savez
    writer(path, **payload)


def write_shard_manifest(
    path: Path,
    *,
    metadata: Mapping[str, object],
    shards: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": SHARD_FORMAT_VERSION,
        "metadata": dict(metadata),
        "n_clips": int(sum(int(shard["n_clips"]) for shard in shards)),
        "n_shards": int(len(shards)),
        "shards": [dict(shard) for shard in shards],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def shard_rows_for_manifest(shards: Sequence[ShardWrite]) -> list[dict[str, object]]:
    return [
        {
            "index": int(shard.index),
            "path": shard.relpath,
            "n_clips": int(shard.n_clips),
            "sample_ids": list(shard.sample_ids),
        }
        for shard in shards
    ]
