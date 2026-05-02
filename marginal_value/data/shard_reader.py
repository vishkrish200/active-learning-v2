from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np


def load_shard_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def iter_shard_arrays(manifest: Mapping[str, object]) -> Iterator[dict[str, np.ndarray]]:
    manifest_path = Path(str(manifest["_manifest_path"]))
    for shard in manifest.get("shards", []):
        if not isinstance(shard, Mapping):
            raise ValueError(f"Malformed shard entry in {manifest_path}")
        shard_path = manifest_path.parent / str(shard["path"])
        with np.load(shard_path, allow_pickle=False) as data:
            yield {key: np.asarray(data[key]).copy() for key in data.files}


def load_representation_matrix(manifest: Mapping[str, object], representation: str) -> tuple[list[str], np.ndarray]:
    sample_ids: list[str] = []
    matrices: list[np.ndarray] = []
    key = f"rep__{representation}"
    for shard in iter_shard_arrays(manifest):
        if key not in shard:
            raise KeyError(f"Shard is missing representation '{representation}'.")
        sample_ids.extend(str(value) for value in shard["sample_ids"].tolist())
        matrices.append(np.asarray(shard[key], dtype="float32"))
    if not matrices:
        return [], np.zeros((0, 0), dtype="float32")
    return sample_ids, np.vstack(matrices).astype("float32")
