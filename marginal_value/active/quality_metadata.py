from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from marginal_value.active.registry import (
    ClipRecord,
    audit_clip_registry_coverage_from_config,
    load_clip_registry_from_config,
)
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import compute_quality_from_jsonl


QUALITY_METADATA_COLUMNS = (
    "sample_id",
    "split",
    "worker_id",
    "source_group_id",
    "url",
    "quality_score",
    "n_samples",
    "duration_sec",
    "missing_rate",
    "nan_fraction",
    "inf_fraction",
    "flatline_fraction",
    "saturation_fraction",
    "max_abs_value",
    "extreme_value_fraction",
    "spike_rate",
    "high_frequency_energy",
    "stationary_fraction",
    "axis_imbalance",
    "repeated_timestamp_fraction",
    "timestamp_jitter_fraction",
)


def build_active_quality_metadata(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_active_quality_metadata_config(config)
    mode = "smoke" if smoke else "full"
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"active_quality_{mode}.jsonl"
    metadata_csv_path = output_dir / f"active_quality_{mode}.csv"
    report_path = output_dir / f"active_quality_report_{mode}.json"
    log_event("active_quality_metadata", "start", mode=mode)

    registry = load_clip_registry_from_config(config)
    registry_coverage = audit_clip_registry_coverage_from_config(config, registry=registry)
    selected = _select_clips(registry.clips, config=config, smoke=smoke)
    sample_rate = float(config.get("quality", {}).get("sample_rate", 30.0))
    max_samples = config.get("quality", {}).get("max_samples_per_clip")
    max_samples = int(max_samples) if max_samples is not None else None
    low_quality_threshold = float(config.get("quality", {}).get("low_quality_threshold", 0.45))

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    progress_every = max(1, len(selected) // 10) if selected else 1
    for index, clip in enumerate(selected, start=1):
        try:
            features = compute_quality_from_jsonl(
                clip.raw_path,
                sample_rate=sample_rate,
                max_samples=max_samples,
            )
            row = _quality_row(clip, features)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch.
            if len(failures) < 25:
                failures.append({"sample_id": clip.sample_id, "error": str(exc)})
        log_progress(
            "active_quality_metadata",
            "quality_progress",
            index=index,
            total=len(selected),
            every=progress_every,
            mode=mode,
            rows_written=len(rows),
            failed=len(failures),
        )

    _write_jsonl(metadata_path, rows)
    _write_csv(metadata_csv_path, rows)
    low_quality_count = sum(float(row.get("quality_score", 1.0)) <= low_quality_threshold for row in rows)
    report = {
        "mode": mode,
        "metadata_path": str(metadata_path),
        "metadata_csv_path": str(metadata_csv_path),
        "selected_clip_count": len(selected),
        "rows_written": len(rows),
        "failed_count": len(failures),
        "failure_examples": failures,
        "low_quality_threshold": low_quality_threshold,
        "low_quality_count": int(low_quality_count),
        "low_quality_fraction": float(low_quality_count / len(rows)) if rows else 0.0,
        "registry_coverage": _deduplicate_coverage_aliases(registry_coverage),
        "registry_coverage_summary": _registry_coverage_summary(registry_coverage),
        "selection": _selection_report(selected),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "metadata_path": str(metadata_path),
        "metadata_csv_path": str(metadata_csv_path),
        "report_path": str(report_path),
        "selected_clip_count": len(selected),
        "rows_written": len(rows),
        "low_quality_count": int(low_quality_count),
        "failed_count": len(failures),
    }
    log_event("active_quality_metadata", "done", **result)
    return result


def validate_active_quality_metadata_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    _required_mapping(config, "quality")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Active quality metadata must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    if not manifests:
        raise ValueError("data.manifests must not be empty.")
    for manifest in manifests.values():
        if not str(manifest).startswith("cache/manifests/"):
            raise ValueError("data.manifests paths must be under cache/manifests/.")
    if int(execution.get("smoke_max_clips_per_split", 1)) <= 0:
        raise ValueError("execution.smoke_max_clips_per_split must be positive.")
    low_quality_threshold = float(config.get("quality", {}).get("low_quality_threshold", 0.45))
    if not 0.0 <= low_quality_threshold <= 1.0:
        raise ValueError("quality.low_quality_threshold must be in [0, 1].")


def _select_clips(clips: Sequence[ClipRecord], *, config: Mapping[str, Any], smoke: bool) -> list[ClipRecord]:
    quality_config = config.get("quality", {})
    if not isinstance(quality_config, Mapping):
        quality_config = {}
    selection = quality_config.get("selection", {})
    if not isinstance(selection, Mapping):
        selection = {}
    split_filter = selection.get("splits")
    splits = {str(split) for split in split_filter} if isinstance(split_filter, list | tuple) else None
    selected = [clip for clip in clips if splits is None or clip.split in splits]
    sort_by = str(selection.get("sort_by", "manifest_order"))
    if sort_by == "sample_id":
        selected = sorted(selected, key=lambda clip: (clip.split, clip.sample_id))
    elif sort_by == "source_group":
        selected = sorted(selected, key=lambda clip: (clip.split, clip.source_group_id, clip.sample_id))
    elif sort_by != "manifest_order":
        raise ValueError("quality.selection.sort_by must be manifest_order, sample_id, or source_group.")

    if smoke:
        max_per_split = int(config.get("execution", {}).get("smoke_max_clips_per_split", 128))
    else:
        max_per_split = selection.get("max_clips_per_split")
        if max_per_split is None:
            return selected
        max_per_split = int(max_per_split)
        if max_per_split <= 0:
            return selected
    counts: dict[str, int] = {}
    limited: list[ClipRecord] = []
    for clip in selected:
        count = counts.get(clip.split, 0)
        if count >= max_per_split:
            continue
        limited.append(clip)
        counts[clip.split] = count + 1
    return limited


def _quality_row(clip: ClipRecord, features: Mapping[str, float]) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": clip.sample_id,
        "split": clip.split,
        "worker_id": clip.worker_id,
        "source_group_id": clip.source_group_id,
        "url": clip.url,
    }
    for key in QUALITY_METADATA_COLUMNS:
        if key in row:
            continue
        row[key] = float(features.get(key, 0.0))
    return row


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QUALITY_METADATA_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _selection_report(selected: Sequence[ClipRecord]) -> dict[str, object]:
    by_split: dict[str, int] = {}
    for clip in selected:
        by_split[clip.split] = by_split.get(clip.split, 0) + 1
    return {
        "selected_clip_count": len(selected),
        "selected_by_split": dict(sorted(by_split.items())),
    }


def _deduplicate_coverage_aliases(coverage: Mapping[str, Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    seen_ids: set[int] = set()
    for key in ("old", "new"):
        if key in coverage:
            output[key] = coverage[key]
            seen_ids.add(id(coverage[key]))
    for key, value in coverage.items():
        if key not in output and id(value) not in seen_ids:
            output[key] = value
    return output


def _registry_coverage_summary(coverage: Mapping[str, Mapping[str, object]]) -> dict[str, int]:
    old = coverage.get("old", {})
    new = coverage.get("new", {})
    return {
        "manifest_old_url_count": int(old.get("manifest_url_count", 0)),
        "cached_old_url_count": int(old.get("cached_url_count", 0)),
        "registry_old_clip_count": int(old.get("registry_clip_count", 0)),
        "unique_old_workers": int(old.get("unique_workers", 0)),
        "unique_old_source_groups": int(old.get("unique_source_groups", 0)),
        "manifest_new_url_count": int(new.get("manifest_url_count", 0)),
        "cached_new_url_count": int(new.get("cached_url_count", 0)),
        "registry_new_clip_count": int(new.get("registry_clip_count", 0)),
        "unique_new_workers": int(new.get("unique_workers", 0)),
        "skipped_uncached_count": int(old.get("skipped_uncached_count", 0)) + int(new.get("skipped_uncached_count", 0)),
        "skipped_missing_raw_count": int(old.get("skipped_missing_raw_count", 0)) + int(new.get("skipped_missing_raw_count", 0)),
        "skipped_missing_feature_count": int(old.get("skipped_missing_feature_count", 0)) + int(new.get("skipped_missing_feature_count", 0)),
    }


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"active quality metadata config requires object field '{key}'.")
    return value
