from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from marginal_value.active.embedding_cache import SUPPORTED_REPRESENTATIONS, load_embedding_lookup
from marginal_value.active.registry import ClipRecord, audit_clip_registry_coverage_from_config, load_clip_registry_from_config
from marginal_value.data.shards import ShardWrite, shard_rows_for_manifest, write_clip_shard, write_shard_manifest
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import load_modal_jsonl_imu


DEFAULT_PROGRESS_EVERY_SHARDS = 1


def run_build_full_support_shards(
    config: dict[str, Any],
    *,
    smoke: bool = False,
    on_shard_written: Callable[[], None] | None = None,
) -> dict[str, Any]:
    validate_build_full_support_shards_config(config)
    mode = "smoke" if smoke else "full"
    start_time = time.time()
    shard_config = config["shards"]
    output_dir = Path(str(shard_config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"full_support_shards_{mode}.json"
    report_path = output_dir / f"full_support_shard_report_{mode}.json"
    progress_path = output_dir / f"full_support_shard_progress_{mode}.jsonl"
    progress = _ProgressLog(progress_path)
    log_event("full_support_shards", "start", mode=mode, output_dir=str(output_dir))
    progress.write("start", mode=mode, output_dir=str(output_dir))

    registry = load_clip_registry_from_config(config)
    registry_coverage = audit_clip_registry_coverage_from_config(config, registry=registry)
    split_names = [str(split) for split in shard_config.get("clip_splits", config["data"]["manifests"].keys())]
    clips, coverage = _clips_for_shards(config, split_names=split_names, smoke=smoke, registry=registry)
    if not clips:
        raise ValueError("No clips selected for full-support shard build.")

    shard_size = int(shard_config.get("shard_size", 2048))
    representations = [str(rep) for rep in shard_config.get("representations", ["window_mean_std_pool"])]
    sample_rate = float(shard_config.get("sample_rate", 30.0))
    raw_shape_max_samples = shard_config.get("raw_shape_max_samples")
    include_imu_samples = bool(shard_config.get("include_imu_samples", False))
    imu_max_samples = int(shard_config.get("imu_max_samples", 5400))
    compress = bool(shard_config.get("compress", False))
    workers = int(shard_config.get("workers", 1))
    shards = _shard_plan(clips, shard_size=shard_size)
    progress_every = int(shard_config.get("progress_every_shards", DEFAULT_PROGRESS_EVERY_SHARDS))
    quality_summary = _quality_summary(clips)
    completed: list[ShardWrite] = []

    selection_payload = {
        "mode": mode,
        "n_clips": len(clips),
        "n_shards": len(shards),
        "representations": representations,
        "quality_score_present_count": quality_summary["quality_score_present_count"],
        "all_quality_fields_present_count": quality_summary["all_quality_fields_present_count"],
    }
    log_event("full_support_shards", "selection_ready", **selection_payload)
    progress.write("selection_ready", **selection_payload)
    if quality_summary["all_quality_fields_present_count"] < len(clips):
        missing_payload = {
            "mode": mode,
            "n_clips": len(clips),
            "missing_any_quality_field_count": quality_summary["missing_any_quality_field_count"],
            "all_quality_fields_present_count": quality_summary["all_quality_fields_present_count"],
        }
        log_event("full_support_shards", "quality_metadata_incomplete", **missing_payload)
        progress.write("quality_metadata_incomplete", **missing_payload)

    _write_report(
        report_path,
        status="running",
        mode=mode,
        coverage=coverage,
        registry_coverage=registry_coverage,
        quality_summary=quality_summary,
        manifest_path=manifest_path,
        progress_path=progress_path,
        started_at=start_time,
        completed=completed,
        total_shards=len(shards),
        total_clips=len(clips),
    )
    progress.write("report_write", status="running", completed_shards=0, total_shards=len(shards))

    for shard_index, shard_clips in enumerate(shards):
        relpath = f"shards_{mode}/shard_{shard_index:05d}.npz"
        shard_path = output_dir / relpath
        progress.write(
            "shard_start",
            shard_index=shard_index,
            total_shards=len(shards),
            n_clips=len(shard_clips),
            shard_path=str(shard_path),
        )
        log_event(
            "full_support_shards",
            "shard_start",
            mode=mode,
            shard_index=shard_index,
            total_shards=len(shards),
            n_clips=len(shard_clips),
        )
        embedding_result = load_embedding_lookup(
            shard_clips,
            representations=representations,
            sample_rate=sample_rate,
            raw_shape_max_samples=raw_shape_max_samples,
            cache_dir=None,
            component="full_support_shards",
            mode=f"{mode}_shard_{shard_index:05d}",
            workers=workers,
            representation_options=_representation_options(config, shard_config),
        )
        matrices = {
            representation: np.vstack(
                [embedding_result.embeddings[representation][clip.sample_id] for clip in shard_clips]
            ).astype("float32")
            for representation in representations
        }
        imu_payload = _imu_payload(shard_clips, max_samples=imu_max_samples) if include_imu_samples else {}
        write_clip_shard(
            shard_path,
            sample_ids=[clip.sample_id for clip in shard_clips],
            worker_ids=[clip.worker_id for clip in shard_clips],
            source_group_ids=[clip.source_group_id for clip in shard_clips],
            splits=[clip.split for clip in shard_clips],
            urls=[clip.url for clip in shard_clips],
            quality=_quality_arrays(shard_clips),
            representations=matrices,
            imu_samples=imu_payload.get("imu_samples"),
            imu_lengths=imu_payload.get("imu_lengths"),
            compress=compress,
        )
        completed.append(
            ShardWrite(
                index=shard_index,
                path=shard_path,
                relpath=relpath,
                n_clips=len(shard_clips),
                sample_ids=[clip.sample_id for clip in shard_clips],
            )
        )
        completed_clips = sum(item.n_clips for item in completed)
        metrics = _progress_metrics(completed_clips=completed_clips, total_clips=len(clips), started_at=start_time)
        progress.write(
            "shard_write",
            shard_index=shard_index,
            total_shards=len(shards),
            n_clips=len(shard_clips),
            shard_path=str(shard_path),
            completed_clips=completed_clips,
            total_clips=len(clips),
            **metrics,
        )
        log_progress(
            "full_support_shards",
            "shard_progress",
            index=len(completed),
            total=len(shards),
            every=progress_every,
            mode=mode,
            completed_clips=completed_clips,
            total_clips=len(clips),
            **metrics,
        )
        _write_report(
            report_path,
            status="running",
            mode=mode,
            coverage=coverage,
            registry_coverage=registry_coverage,
            quality_summary=quality_summary,
            manifest_path=manifest_path,
            progress_path=progress_path,
            started_at=start_time,
            completed=completed,
            total_shards=len(shards),
            total_clips=len(clips),
        )
        progress.write(
            "report_write",
            status="running",
            completed_shards=len(completed),
            total_shards=len(shards),
            completed_clips=completed_clips,
            total_clips=len(clips),
            **metrics,
        )
        if on_shard_written is not None:
            on_shard_written()

    metadata = {
        "mode": mode,
        "split_names": split_names,
        "representations": representations,
        "sample_rate": sample_rate,
        "raw_shape_max_samples": None if raw_shape_max_samples is None else int(raw_shape_max_samples),
        "include_imu_samples": include_imu_samples,
        "imu_max_samples": imu_max_samples if include_imu_samples else None,
        "compress": compress,
        "shard_size": shard_size,
    }
    write_shard_manifest(manifest_path, metadata=metadata, shards=shard_rows_for_manifest(completed))
    _write_report(
        report_path,
        status="done",
        mode=mode,
        coverage=coverage,
        registry_coverage=registry_coverage,
        quality_summary=quality_summary,
        manifest_path=manifest_path,
        progress_path=progress_path,
        started_at=start_time,
        completed=completed,
        total_shards=len(shards),
        total_clips=len(clips),
    )
    progress.write("manifest_write", manifest_path=str(manifest_path), n_shards=len(shards), n_clips=len(clips))
    progress.write("done", report_path=str(report_path), manifest_path=str(manifest_path), n_shards=len(shards), n_clips=len(clips))
    result = {
        "mode": mode,
        "n_clips": len(clips),
        "n_shards": len(shards),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "progress_path": str(progress_path),
    }
    log_event("full_support_shards", "done", **result)
    return result


def validate_build_full_support_shards_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    shards = _required_mapping(config, "shards")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Full-support shard build must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(shards.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("shards.output_dir must be mounted under /artifacts.")
    if not manifests:
        raise ValueError("data.manifests must not be empty.")
    split_names = shards.get("clip_splits", list(manifests))
    if not isinstance(split_names, list | tuple) or not split_names:
        raise ValueError("shards.clip_splits must be a non-empty list.")
    missing = sorted({str(split) for split in split_names} - set(str(split) for split in manifests))
    if missing:
        raise ValueError(f"shards.clip_splits references unknown manifest splits: {missing}")
    representations = shards.get("representations", ["window_mean_std_pool"])
    if not isinstance(representations, list | tuple) or not representations:
        raise ValueError("shards.representations must be a non-empty list.")
    unsupported = set(str(rep) for rep in representations) - set(SUPPORTED_REPRESENTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported shard representations: {sorted(unsupported)}")
    if int(shards.get("shard_size", 2048)) <= 0:
        raise ValueError("shards.shard_size must be positive.")
    if int(shards.get("workers", 1)) <= 0:
        raise ValueError("shards.workers must be positive.")
    if int(shards.get("progress_every_shards", DEFAULT_PROGRESS_EVERY_SHARDS)) <= 0:
        raise ValueError("shards.progress_every_shards must be positive.")
    if shards.get("max_clips_per_split") is not None and int(shards["max_clips_per_split"]) <= 0:
        raise ValueError("shards.max_clips_per_split must be positive when provided.")
    if int(execution.get("smoke_max_clips_per_split", 32)) <= 0:
        raise ValueError("execution.smoke_max_clips_per_split must be positive.")


def _clips_for_shards(config: Mapping[str, Any], *, split_names: Sequence[str], smoke: bool, registry) -> tuple[list[ClipRecord], dict[str, object]]:
    shard_config = config["shards"]
    selected: list[ClipRecord] = []
    coverage: dict[str, object] = {}
    smoke_cap = int(config["execution"].get("smoke_max_clips_per_split", 32))
    full_cap = shard_config.get("max_clips_per_split")
    fail_if_exceeds = bool(shard_config.get("fail_if_split_exceeds_max", False))
    for split in split_names:
        split_clips = sorted(registry.clips_for_split(str(split)), key=lambda clip: clip.sample_id)
        total = len(split_clips)
        if smoke:
            split_clips = split_clips[:smoke_cap]
        elif full_cap is not None:
            cap = int(full_cap)
            if total > cap and fail_if_exceeds:
                raise ValueError(f"Split '{split}' has {total} clips, exceeding shards.max_clips_per_split={cap}.")
            split_clips = split_clips[:cap]
        coverage[str(split)] = {
            "available_count": int(total),
            "selected_count": int(len(split_clips)),
            "selected_fraction": _fraction(len(split_clips), total),
        }
        selected.extend(split_clips)
    return selected, coverage


def _quality_arrays(clips: Sequence[ClipRecord]) -> dict[str, np.ndarray]:
    return {
        "quality_score": np.asarray([_quality_value(clip, "quality_score") for clip in clips], dtype="float32"),
        "stationary_fraction": np.asarray([_quality_value(clip, "stationary_fraction") for clip in clips], dtype="float32"),
        "max_abs_value": np.asarray([_quality_value(clip, "max_abs_value") for clip in clips], dtype="float32"),
    }


def _quality_value(clip: ClipRecord, key: str) -> float:
    if key not in clip.quality:
        return float("nan")
    return float(clip.quality[key])


def _quality_summary(clips: Sequence[ClipRecord]) -> dict[str, int]:
    keys = ("quality_score", "stationary_fraction", "max_abs_value")
    arrays = {key: np.asarray([_quality_value(clip, key) for clip in clips], dtype="float32") for key in keys}
    present_by_key = {key: int(np.isfinite(values).sum()) for key, values in arrays.items()}
    all_present = np.ones((len(clips),), dtype=bool)
    for values in arrays.values():
        all_present &= np.isfinite(values)
    return {
        "quality_score_present_count": present_by_key["quality_score"],
        "stationary_fraction_present_count": present_by_key["stationary_fraction"],
        "max_abs_value_present_count": present_by_key["max_abs_value"],
        "all_quality_fields_present_count": int(all_present.sum()),
        "missing_any_quality_field_count": int(len(clips) - all_present.sum()),
    }


def _imu_payload(clips: Sequence[ClipRecord], *, max_samples: int) -> dict[str, np.ndarray]:
    samples = np.zeros((len(clips), int(max_samples), 6), dtype="float32")
    lengths = np.zeros((len(clips),), dtype="int32")
    for index, clip in enumerate(clips):
        values, _timestamps = load_modal_jsonl_imu(clip.raw_path, max_samples=int(max_samples))
        values = _normalize_imu(values)
        keep = min(len(values), int(max_samples))
        if keep:
            samples[index, :keep] = values[:keep].astype("float32")
        lengths[index] = int(keep)
    return {"imu_samples": samples, "imu_lengths": lengths}


def _normalize_imu(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="float32")
    mean = float(np.mean(array)) if array.size else 0.0
    std = float(np.std(array)) if array.size else 1.0
    if not np.isfinite(std) or std < 1.0e-6:
        std = 1.0
    return (array - mean) / std


def _shard_plan(clips: Sequence[ClipRecord], *, shard_size: int) -> list[list[ClipRecord]]:
    return [list(clips[start : start + int(shard_size)]) for start in range(0, len(clips), int(shard_size))]


def _write_report(
    path: Path,
    *,
    status: str,
    mode: str,
    coverage: Mapping[str, object],
    registry_coverage: Mapping[str, object],
    quality_summary: Mapping[str, int],
    manifest_path: Path,
    progress_path: Path,
    started_at: float,
    completed: Sequence[ShardWrite],
    total_shards: int,
    total_clips: int,
) -> None:
    completed_clips = sum(item.n_clips for item in completed)
    report = {
        "status": status,
        "mode": mode,
        "n_clips": int(total_clips),
        "n_shards": int(total_shards),
        "completed_shards": int(len(completed)),
        "completed_clips": int(completed_clips),
        "completed_fraction": _fraction(completed_clips, total_clips),
        "elapsed_seconds": float(time.time() - started_at),
        "coverage": coverage,
        "registry_coverage": registry_coverage,
        "quality_metadata": dict(quality_summary),
        "artifacts": {
            "manifest": str(manifest_path),
            "progress": str(progress_path),
            "report": str(path),
        },
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


class _ProgressLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, **fields: object) -> None:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "component": "full_support_shards",
            "event": event,
        }
        payload.update(fields)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _progress_metrics(*, completed_clips: int, total_clips: int, started_at: float) -> dict[str, float | None]:
    elapsed = max(time.time() - started_at, 0.0)
    completed_fraction = _fraction(completed_clips, total_clips)
    clips_per_second = float(completed_clips / elapsed) if elapsed > 0 and completed_clips > 0 else None
    remaining = max(total_clips - completed_clips, 0)
    eta_seconds = float(remaining / clips_per_second) if clips_per_second else None
    return {
        "completed_fraction": completed_fraction,
        "elapsed_seconds": float(elapsed),
        "clips_per_second": clips_per_second,
        "eta_seconds": eta_seconds,
    }


def _representation_options(config: Mapping[str, Any], section: Mapping[str, Any]) -> Mapping[str, object]:
    options: dict[str, object] = {}
    embeddings = config.get("embeddings")
    if isinstance(embeddings, Mapping):
        options.update(dict(embeddings))
    explicit = section.get("representation_options")
    if isinstance(explicit, Mapping):
        options.update(dict(explicit))
    return options


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Full-support shard config requires object field '{key}'.")
    return value


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
