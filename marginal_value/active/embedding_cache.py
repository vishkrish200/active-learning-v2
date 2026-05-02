from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from marginal_value.active.registry import ClipRecord
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import load_modal_jsonl_imu
from marginal_value.models.ts2vec_inference import TS2VecInference
from marginal_value.ranking.baseline_ranker import (
    raw_shape_stats_embedding,
    temporal_order_embedding,
    window_mean_std_embedding,
)


EMBEDDING_CACHE_VERSION = "active_embedding_cache_v1"
SUPPORTED_REPRESENTATIONS = (
    "window_mean_std_pool",
    "temporal_order",
    "raw_shape_stats",
    "window_shape_stats",
    "ts2vec",
    "ts2vec_multiscale",
)


@dataclass(frozen=True)
class EmbeddingLookupResult:
    embeddings: dict[str, dict[str, np.ndarray]]
    cache_status: str
    cache_path: Path | None
    n_clips: int

    def report(self) -> dict[str, object]:
        return {
            "status": self.cache_status,
            "path": str(self.cache_path) if self.cache_path is not None else "",
            "n_clips": int(self.n_clips),
        }


def load_embedding_lookup(
    clips: Sequence[ClipRecord],
    *,
    representations: Sequence[str],
    sample_rate: float,
    raw_shape_max_samples: object | None,
    cache_dir: str | Path | None = None,
    component: str = "active_embedding_cache",
    mode: str = "full",
    representation_options: Mapping[str, object] | None = None,
    workers: int = 1,
) -> EmbeddingLookupResult:
    reps = [str(rep) for rep in representations]
    _validate_representations(reps)
    ordered_clips = sorted(clips, key=lambda clip: clip.sample_id)
    metadata = _cache_metadata(
        ordered_clips,
        representations=reps,
        sample_rate=sample_rate,
        raw_shape_max_samples=raw_shape_max_samples,
        representation_options=representation_options,
    )
    cache_path = _cache_path(cache_dir, metadata)
    if cache_path is not None and cache_path.exists():
        log_event(component, "embedding_cache_hit", mode=mode, cache_path=str(cache_path), n_clips=len(ordered_clips))
        return EmbeddingLookupResult(
            embeddings=_read_cache_pack(cache_path, metadata),
            cache_status="hit",
            cache_path=cache_path,
            n_clips=len(ordered_clips),
        )
    shard_manifest_path = _shard_manifest_path(cache_dir, metadata)
    if shard_manifest_path is not None and shard_manifest_path.exists():
        log_event(
            component,
            "embedding_shard_cache_hit",
            mode=mode,
            cache_path=str(shard_manifest_path),
            n_clips=len(ordered_clips),
        )
        return EmbeddingLookupResult(
            embeddings=_read_shard_cache(shard_manifest_path, metadata),
            cache_status="shard_hit",
            cache_path=shard_manifest_path,
            n_clips=len(ordered_clips),
        )

    log_event(
        component,
        "embedding_cache_miss" if cache_path is not None else "embedding_cache_disabled",
        mode=mode,
        cache_path=str(cache_path) if cache_path is not None else "",
        n_clips=len(ordered_clips),
    )
    matrices = _compute_embedding_matrices(
        ordered_clips,
        representations=reps,
        sample_rate=sample_rate,
        raw_shape_max_samples=raw_shape_max_samples,
        component=component,
        mode=mode,
        workers=int(workers),
        representation_options=representation_options,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_cache_pack(cache_path, metadata, matrices)
        log_event(component, "embedding_cache_write", mode=mode, cache_path=str(cache_path), n_clips=len(ordered_clips))
    return EmbeddingLookupResult(
        embeddings=_matrices_to_lookup(metadata["sample_ids"], matrices),
        cache_status="miss" if cache_path is not None else "disabled",
        cache_path=cache_path,
        n_clips=len(ordered_clips),
    )


def write_embedding_shard_cache(
    clips: Sequence[ClipRecord],
    *,
    representations: Sequence[str],
    sample_rate: float,
    raw_shape_max_samples: object | None,
    cache_dir: str | Path,
    shard_size: int,
    component: str = "active_embedding_cache",
    mode: str = "full",
    on_shard_written: Callable[[], None] | None = None,
    workers: int = 1,
    representation_options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    reps = [str(rep) for rep in representations]
    _validate_representations(reps)
    if int(shard_size) <= 0:
        raise ValueError("shard_size must be positive.")
    if int(workers) <= 0:
        raise ValueError("workers must be positive.")
    ordered_clips = sorted(clips, key=lambda clip: clip.sample_id)
    metadata = _cache_metadata(
        ordered_clips,
        representations=reps,
        sample_rate=sample_rate,
        raw_shape_max_samples=raw_shape_max_samples,
        representation_options=representation_options,
    )
    manifest_path = _shard_manifest_path(cache_dir, metadata)
    if manifest_path is None:
        raise ValueError("cache_dir is required for sharded embedding precompute.")
    shard_dir = _shard_dir(cache_dir, metadata)
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    shards = _shard_plan(ordered_clips, shard_size=int(shard_size))
    manifest_shards: list[dict[str, object]] = []
    for shard_index, shard_clips in enumerate(shards):
        relpath = f"{shard_dir.name}/shard_{shard_index:05d}.npz"
        shard_path = manifest_path.parent / relpath
        sample_ids = [clip.sample_id for clip in shard_clips]
        if shard_path.exists():
            log_event(
                component,
                "embedding_shard_cache_skip_existing",
                mode=mode,
                shard_index=shard_index,
                n_shards=len(shards),
                shard_path=str(shard_path),
                n_clips=len(shard_clips),
            )
        else:
            matrices = _compute_embedding_matrices(
                shard_clips,
                representations=reps,
                sample_rate=sample_rate,
                raw_shape_max_samples=raw_shape_max_samples,
                component=component,
                mode=mode,
                workers=int(workers),
                representation_options=representation_options,
            )
            _write_shard_file(shard_path, sample_ids, matrices)
            log_event(
                component,
                "embedding_shard_cache_write",
                mode=mode,
                shard_index=shard_index + 1,
                n_shards=len(shards),
                shard_path=str(shard_path),
                n_clips=len(shard_clips),
            )
            if on_shard_written is not None:
                on_shard_written()
        manifest_shards.append(
            {
                "index": int(shard_index),
                "path": relpath,
                "sample_ids": sample_ids,
                "n_clips": int(len(shard_clips)),
            }
        )

    manifest = {
        "metadata": metadata,
        "n_clips": int(len(ordered_clips)),
        "n_shards": int(len(shards)),
        "shards": manifest_shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log_event(
        component,
        "embedding_shard_manifest_write",
        mode=mode,
        manifest_path=str(manifest_path),
        n_clips=len(ordered_clips),
        n_shards=len(shards),
    )
    return {
        "manifest_path": str(manifest_path),
        "n_clips": int(len(ordered_clips)),
        "n_shards": int(len(shards)),
        "shard_dir": str(shard_dir),
    }


def embedding_cache_dir_from_config(config: Mapping[str, object]) -> Path | None:
    value = config.get("embeddings")
    if not isinstance(value, Mapping):
        return None
    if value.get("enabled", True) is False:
        return None
    cache_dir = value.get("cache_dir")
    if cache_dir is None or str(cache_dir).strip() == "":
        return None
    return Path(str(cache_dir))


def _compute_embedding_matrices(
    clips: Sequence[ClipRecord],
    *,
    representations: Sequence[str],
    sample_rate: float,
    raw_shape_max_samples: object | None,
    component: str,
    mode: str,
    workers: int = 1,
    representation_options: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {rep: [] for rep in representations}
    max_samples = int(raw_shape_max_samples) if raw_shape_max_samples is not None else None
    ts2vec_inference = _build_ts2vec_inference(representations, representation_options or {})
    if ts2vec_inference is not None:
        return _compute_embedding_matrices_with_ts2vec(
            clips,
            representations=representations,
            sample_rate=sample_rate,
            max_samples=max_samples,
            component=component,
            mode=mode,
            ts2vec_inference=ts2vec_inference,
            representation_options=representation_options or {},
        )
    progress_every = max(1, len(clips) // 10) if clips else 1
    worker_count = min(int(workers), len(clips)) if clips else 1
    if worker_count <= 1:
        row_iter = (
            _compute_clip_embedding_rows(
                clip,
                representations=representations,
                sample_rate=sample_rate,
                max_samples=max_samples,
                ts2vec_inference=ts2vec_inference,
                representation_options=representation_options or {},
            )
            for clip in clips
        )
        close_pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=worker_count)
        close_pool = pool
        row_iter = pool.map(
            lambda clip: _compute_clip_embedding_rows(
                clip,
                representations=representations,
                sample_rate=sample_rate,
                max_samples=max_samples,
                ts2vec_inference=ts2vec_inference,
                representation_options=representation_options or {},
            ),
            clips,
        )
    try:
        for index, clip_rows in enumerate(row_iter, start=1):
            for representation in representations:
                rows[representation].append(clip_rows[representation])
            log_progress(
                component,
                "embedding_progress",
                index=index,
                total=len(clips),
                every=progress_every,
                mode=mode,
            )
    finally:
        if close_pool is not None:
            close_pool.shutdown(wait=True)
    return {rep: np.vstack(values).astype("float32") for rep, values in rows.items()}


def _compute_embedding_matrices_with_ts2vec(
    clips: Sequence[ClipRecord],
    *,
    representations: Sequence[str],
    sample_rate: float,
    max_samples: int | None,
    component: str,
    mode: str,
    ts2vec_inference: object,
    representation_options: Mapping[str, object],
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {rep: [] for rep in representations if rep != "ts2vec"}
    ts2vec_samples: list[np.ndarray] = []
    non_ts2vec_representations = [rep for rep in representations if rep != "ts2vec"]
    progress_every = max(1, len(clips) // 10) if clips else 1
    for index, clip in enumerate(clips, start=1):
        if non_ts2vec_representations:
            clip_rows = _compute_clip_embedding_rows(
                clip,
                representations=non_ts2vec_representations,
                sample_rate=sample_rate,
                max_samples=max_samples,
                ts2vec_inference=ts2vec_inference,
                representation_options=representation_options,
            )
            for representation in non_ts2vec_representations:
                rows[representation].append(clip_rows[representation])
        if "ts2vec" in representations:
            ts2vec_max_samples = representation_options.get("ts2vec_max_samples")
            samples, _timestamps = load_modal_jsonl_imu(
                clip.raw_path,
                max_samples=None if ts2vec_max_samples is None else int(ts2vec_max_samples),
            )
            ts2vec_samples.append(samples)
        log_progress(
            component,
            "embedding_progress",
            index=index,
            total=len(clips),
            every=progress_every,
            mode=mode,
        )
    if "ts2vec" in representations:
        batch_size = int(representation_options.get("ts2vec_batch_size", 32))
        try:
            ts2vec_matrix = ts2vec_inference.encode_batch(ts2vec_samples, batch_size=batch_size)
        except (AttributeError, TypeError):
            ts2vec_matrix = np.vstack([ts2vec_inference.encode_clip(samples) for samples in ts2vec_samples]).astype("float32")
        rows["ts2vec"] = [np.asarray(row, dtype="float32") for row in ts2vec_matrix]
        log_event(
            component,
            "ts2vec_batch_encode_done",
            mode=mode,
            n_clips=len(ts2vec_samples),
            batch_size=batch_size,
        )
    return {rep: np.vstack(rows[rep]).astype("float32") for rep in representations}


def _compute_clip_embedding_rows(
    clip: ClipRecord,
    *,
    representations: Sequence[str],
    sample_rate: float,
    max_samples: int | None,
    ts2vec_inference: object | None = None,
    representation_options: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    windows: np.ndarray | None = None
    raw_shape_samples: np.ndarray | None = None
    ts2vec_samples: np.ndarray | None = None
    options = representation_options or {}
    for representation in representations:
        if representation in {"window_mean_std_pool", "temporal_order", "window_shape_stats"}:
            if windows is None:
                windows = _load_windows(clip)
            if representation == "window_mean_std_pool":
                rows[representation] = window_mean_std_embedding(windows)
            elif representation == "temporal_order":
                rows[representation] = temporal_order_embedding(windows)
            elif representation == "window_shape_stats":
                rows[representation] = _window_shape_stats_embedding(windows)
        elif representation == "raw_shape_stats":
            if raw_shape_samples is None:
                raw_shape_samples, _timestamps = load_modal_jsonl_imu(clip.raw_path, max_samples=max_samples)
            rows[representation] = raw_shape_stats_embedding(raw_shape_samples, sample_rate=sample_rate)
        elif representation in {"ts2vec", "ts2vec_multiscale"}:
            if ts2vec_inference is None:
                raise ValueError("TS2Vec representation requires a ts2vec checkpoint path.")
            if ts2vec_samples is None:
                ts2vec_max_samples = options.get("ts2vec_max_samples")
                ts2vec_samples, _timestamps = load_modal_jsonl_imu(
                    clip.raw_path,
                    max_samples=None if ts2vec_max_samples is None else int(ts2vec_max_samples),
                )
            if representation == "ts2vec":
                rows[representation] = ts2vec_inference.encode_clip(ts2vec_samples)
            else:
                window_sizes = options.get("ts2vec_multiscale_window_sizes", [900, 2700])
                rows[representation] = ts2vec_inference.encode_clip_multiscale(
                    ts2vec_samples,
                    window_sizes=[int(value) for value in window_sizes],
                    stride_ratio=float(options.get("ts2vec_multiscale_stride_ratio", 0.5)),
                    pool=str(options.get("ts2vec_multiscale_pool", "max")),
                )
        else:
            raise ValueError(f"Unsupported representation: {representation}")
    return rows


def _write_cache_pack(path: Path, metadata: Mapping[str, object], matrices: Mapping[str, np.ndarray]) -> None:
    payload: dict[str, object] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "sample_ids": np.asarray(metadata["sample_ids"], dtype=str),
    }
    for representation, matrix in matrices.items():
        payload[f"rep__{representation}"] = np.asarray(matrix, dtype="float32")
    np.savez_compressed(path, **payload)


def _read_cache_pack(path: Path, expected_metadata: Mapping[str, object]) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata != expected_metadata:
            raise ValueError(f"Embedding cache metadata mismatch for {path}")
        sample_ids = [str(sample_id) for sample_id in data["sample_ids"].tolist()]
        matrices = {
            representation: np.asarray(data[f"rep__{representation}"], dtype="float32").copy()
            for representation in expected_metadata["representations"]
        }
    return _matrices_to_lookup(sample_ids, matrices)


def _write_shard_file(path: Path, sample_ids: Sequence[str], matrices: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"sample_ids": np.asarray(sample_ids, dtype=str)}
    for representation, matrix in matrices.items():
        payload[f"rep__{representation}"] = np.asarray(matrix, dtype="float32")
    np.savez(path, **payload)


def _read_shard_cache(
    manifest_path: Path,
    expected_metadata: Mapping[str, object],
) -> dict[str, dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata")
    if metadata != expected_metadata:
        raise ValueError(f"Embedding shard cache metadata mismatch for {manifest_path}")
    sample_ids: list[str] = []
    rows: dict[str, list[np.ndarray]] = {
        str(rep): []
        for rep in expected_metadata["representations"]
    }
    for shard in manifest.get("shards", []):
        if not isinstance(shard, Mapping):
            raise ValueError(f"Malformed shard entry in {manifest_path}")
        shard_path = manifest_path.parent / str(shard["path"])
        with np.load(shard_path, allow_pickle=False) as data:
            shard_ids = [str(sample_id) for sample_id in data["sample_ids"].tolist()]
            sample_ids.extend(shard_ids)
            for representation in rows:
                rows[representation].append(np.asarray(data[f"rep__{representation}"], dtype="float32").copy())
    matrices = {
        representation: np.vstack(parts).astype("float32")
        for representation, parts in rows.items()
    }
    if sample_ids != list(expected_metadata["sample_ids"]):
        raise ValueError(f"Embedding shard cache sample order mismatch for {manifest_path}")
    return _matrices_to_lookup(sample_ids, matrices)


def _matrices_to_lookup(
    sample_ids: Sequence[str],
    matrices: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        representation: {
            str(sample_id): np.asarray(matrix[index], dtype="float32")
            for index, sample_id in enumerate(sample_ids)
        }
        for representation, matrix in matrices.items()
    }


def _cache_metadata(
    clips: Sequence[ClipRecord],
    *,
    representations: Sequence[str],
    sample_rate: float,
    raw_shape_max_samples: object | None,
    representation_options: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "version": EMBEDDING_CACHE_VERSION,
        "sample_ids": [clip.sample_id for clip in clips],
        "representations": [str(rep) for rep in representations],
        "sample_rate": float(sample_rate),
        "raw_shape_max_samples": int(raw_shape_max_samples) if raw_shape_max_samples is not None else None,
        "representation_options": _cacheable_representation_options(representations, representation_options or {}),
    }


def _cache_path(cache_dir: str | Path | None, metadata: Mapping[str, object]) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"embeddings_{digest}.npz"


def _shard_manifest_path(cache_dir: str | Path | None, metadata: Mapping[str, object]) -> Path | None:
    if cache_dir is None:
        return None
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"embeddings_{digest}.shards.json"


def _shard_dir(cache_dir: str | Path, metadata: Mapping[str, object]) -> Path:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"embeddings_{digest}_shards"


def _shard_plan(clips: Sequence[ClipRecord], *, shard_size: int) -> list[list[ClipRecord]]:
    return [
        list(clips[start : start + int(shard_size)])
        for start in range(0, len(clips), int(shard_size))
    ]


def _validate_representations(representations: Sequence[str]) -> None:
    if not representations:
        raise ValueError("At least one embedding representation is required.")
    unsupported = set(str(rep) for rep in representations) - set(SUPPORTED_REPRESENTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported active embedding representations: {sorted(unsupported)}")


def _build_ts2vec_inference(
    representations: Sequence[str],
    options: Mapping[str, object],
) -> TS2VecInference | None:
    if not any(str(rep) in {"ts2vec", "ts2vec_multiscale"} for rep in representations):
        return None
    checkpoint_path = (
        options.get("ts2vec_checkpoint_path")
        or options.get("checkpoint_path")
        or options.get("ts2vec_checkpoint")
    )
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        raise ValueError("TS2Vec representation requires representation_options.ts2vec_checkpoint_path.")
    return TS2VecInference(str(checkpoint_path), device=str(options.get("ts2vec_device", "cpu")))


def _cacheable_representation_options(
    representations: Sequence[str],
    options: Mapping[str, object],
) -> dict[str, object]:
    if not any(str(rep) in {"ts2vec", "ts2vec_multiscale"} for rep in representations):
        return {}
    return {
        "ts2vec_checkpoint_path": str(
            options.get("ts2vec_checkpoint_path")
            or options.get("checkpoint_path")
            or options.get("ts2vec_checkpoint")
            or ""
        ),
        "ts2vec_multiscale_window_sizes": [
            int(value)
            for value in options.get("ts2vec_multiscale_window_sizes", [900, 2700])
        ],
        "ts2vec_multiscale_stride_ratio": float(options.get("ts2vec_multiscale_stride_ratio", 0.5)),
        "ts2vec_multiscale_pool": str(options.get("ts2vec_multiscale_pool", "max")),
        "ts2vec_max_samples": None
        if options.get("ts2vec_max_samples") is None
        else int(options.get("ts2vec_max_samples")),
    }


def _load_windows(clip: ClipRecord) -> np.ndarray:
    with np.load(clip.feature_path) as data:
        return np.asarray(data["window_features"], dtype="float32")


def _window_shape_stats_embedding(windows: np.ndarray) -> np.ndarray:
    values = _finite_2d(windows)
    if len(values) > 1:
        diffs = np.diff(values, axis=0)
        diff_parts = [np.mean(np.abs(diffs), axis=0), np.max(np.abs(diffs), axis=0)]
    else:
        diff_parts = [np.zeros(values.shape[1]), np.zeros(values.shape[1])]
    return np.nan_to_num(
        np.concatenate(
            [
                np.mean(values, axis=0),
                np.std(values, axis=0),
                np.min(values, axis=0),
                np.max(values, axis=0),
                np.percentile(values, 25, axis=0),
                np.percentile(values, 75, axis=0),
                *diff_parts,
            ]
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _finite_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array.")
    if array.shape[0] == 0:
        raise ValueError("Expected at least one row.")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
