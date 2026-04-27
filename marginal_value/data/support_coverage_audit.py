from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from marginal_value.data.cache_support_split import _source_path_for_url, _worker_id_from_url
from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls
from marginal_value.logging_utils import log_event, log_progress


def load_support_coverage_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_support_coverage_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    target = _required_mapping(config, "target")
    artifacts = _required_mapping(config, "artifacts")
    manifests = _required_mapping(target, "manifests")
    source = config.get("source", {})
    expected = config.get("expected", {})
    if execution.get("provider") != "modal":
        raise ValueError("Support coverage audit must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(target.get("root", "")).startswith("/data"):
        raise ValueError("Support coverage target.root must be mounted under /data.")
    if source and not allow_local_paths and not str(source.get("root", "")).startswith("/source"):
        raise ValueError("Support coverage source.root must be mounted under /source when provided.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("Support coverage artifacts.output_dir must be mounted under /artifacts.")
    if not manifests:
        raise ValueError("Support coverage target.manifests must not be empty.")
    for split, manifest in manifests.items():
        if not str(split):
            raise ValueError("Support coverage manifest split names must be non-empty.")
        if not str(manifest).startswith("cache/manifests/"):
            raise ValueError("Support coverage manifest paths must be under cache/manifests/.")
    if "smoke_manifest_samples" in execution and int(execution["smoke_manifest_samples"]) <= 0:
        raise ValueError("execution.smoke_manifest_samples must be positive when provided.")
    if "max_feature_files_for_window_stats" in target and int(target["max_feature_files_for_window_stats"]) <= 0:
        raise ValueError("target.max_feature_files_for_window_stats must be positive when provided.")
    if expected is not None and not isinstance(expected, dict):
        raise ValueError("Support coverage expected section must be an object when provided.")


def run_support_coverage_audit(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_support_coverage_config(config)
    mode = "smoke" if smoke else "full"
    target_root = Path(config["target"]["root"])
    source_root = Path(config.get("source", {}).get("root", "/source"))
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("support_coverage_audit", "start", mode=mode, target_root=str(target_root), source_root=str(source_root))

    feature_by_id = _files_by_stem(target_root, str(config["target"].get("feature_glob", "cache/features/*.npz")))
    raw_by_id = _files_by_stem(target_root, str(config["target"].get("raw_glob", "cache/raw/*.jsonl")))
    manifest_config = config["target"]["manifests"]
    source_splits = config.get("source", {}).get("split_by_manifest", {})
    if source_splits is None:
        source_splits = {}
    if not isinstance(source_splits, dict):
        raise ValueError("source.split_by_manifest must be an object when provided.")

    split_reports: dict[str, Any] = {}
    inspected_ids: set[str] = set()
    for split, manifest_relpath in sorted(manifest_config.items()):
        manifest_path = target_root / str(manifest_relpath)
        urls = read_manifest_urls(manifest_path)
        source_split = str(source_splits.get(split, split))
        report = _audit_manifest_split(
            split=str(split),
            urls=urls,
            source_root=source_root,
            source_split=source_split,
            feature_by_id=feature_by_id,
            raw_by_id=raw_by_id,
            target=config["target"],
            smoke=smoke,
            smoke_samples=int(config["execution"].get("smoke_manifest_samples", 128)),
        )
        split_reports[str(split)] = report
        inspected_ids.update(report["inspected_sample_ids"])
        log_event("support_coverage_audit", "split_done", split=str(split), **_loggable_split_report(report))

    manifest_ids = {
        sample_id
        for split_report in split_reports.values()
        for sample_id in split_report["all_sample_ids"]
    }
    orphan_feature_ids = sorted(set(feature_by_id) - manifest_ids)
    orphan_raw_ids = sorted(set(raw_by_id) - manifest_ids)
    expected_summary = _expected_summary(config.get("expected", {}), split_reports)
    inventory = {
        "feature_file_count": len(feature_by_id),
        "raw_file_count": len(raw_by_id),
        "feature_raw_intersection_count": len(set(feature_by_id) & set(raw_by_id)),
        "orphan_feature_count": len(orphan_feature_ids),
        "orphan_raw_count": len(orphan_raw_ids),
        "orphan_feature_sample_ids": orphan_feature_ids[:25],
        "orphan_raw_sample_ids": orphan_raw_ids[:25],
    }
    report = {
        "mode": mode,
        "target_root": str(target_root),
        "source_root": str(source_root),
        "splits": {
            split: _drop_internal_ids(split_report)
            for split, split_report in split_reports.items()
        },
        "inventory": inventory,
        "expected": expected_summary,
        "notes": [
            "cached_both_count means both raw JSONL and feature NPZ are present for the manifest URL.",
            "feature_window_count counts local 10s feature windows, not the plan's 180s support clips.",
            "source_exists_count checks whether manifest URLs are present in the mounted source mirror.",
        ],
    }
    report_path = output_dir / f"support_coverage_audit_{mode}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "report_path": str(report_path),
        "feature_file_count": inventory["feature_file_count"],
        "raw_file_count": inventory["raw_file_count"],
        "pretrain_cached_both_count": report["splits"].get("pretrain", {}).get("cached_both_count", 0),
        "new_cached_both_count": report["splits"].get("new", {}).get("cached_both_count", 0),
        "pretrain_manifest_url_count": report["splits"].get("pretrain", {}).get("manifest_url_count", 0),
        "new_manifest_url_count": report["splits"].get("new", {}).get("manifest_url_count", 0),
    }
    log_event("support_coverage_audit", "done", **result)
    return result


def _audit_manifest_split(
    *,
    split: str,
    urls: list[str],
    source_root: Path,
    source_split: str,
    feature_by_id: dict[str, Path],
    raw_by_id: dict[str, Path],
    target: dict[str, Any],
    smoke: bool,
    smoke_samples: int,
) -> dict[str, Any]:
    sample_ids = [hash_manifest_url(url) for url in urls]
    inspect_count = min(len(urls), smoke_samples) if smoke else len(urls)
    inspected_urls = urls[:inspect_count]
    inspected_ids = sample_ids[:inspect_count]
    duplicate_urls = len(urls) - len(set(urls))
    duplicate_ids = len(sample_ids) - len(set(sample_ids))
    worker_counts = Counter(_worker_id_from_url(url) for url in urls)

    raw_present_ids = [sample_id for sample_id in inspected_ids if sample_id in raw_by_id]
    feature_present_ids = [sample_id for sample_id in inspected_ids if sample_id in feature_by_id]
    cached_both_ids = [sample_id for sample_id in inspected_ids if sample_id in raw_by_id and sample_id in feature_by_id]

    source_exists_count = 0
    source_check_enabled = source_root.exists()
    progress_every = max(1, inspect_count // 10) if inspect_count else 1
    for index, url in enumerate(inspected_urls, start=1):
        if source_check_enabled and _source_path_for_url(source_root, split=source_split, url=url).exists():
            source_exists_count += 1
        log_progress(
            "support_coverage_audit",
            "source_check_progress",
            index=index,
            total=inspect_count,
            every=progress_every,
            split=split,
            source_exists_count=source_exists_count,
        )

    max_feature_files = target.get("max_feature_files_for_window_stats")
    if max_feature_files is not None:
        max_feature_files = int(max_feature_files)
    feature_paths = [feature_by_id[sample_id] for sample_id in cached_both_ids if sample_id in feature_by_id]
    window_summary = _feature_window_summary(feature_paths, max_files=max_feature_files)
    worker_clip_counts = list(worker_counts.values())
    return {
        "split": split,
        "source_split": source_split,
        "manifest_url_count": len(urls),
        "manifest_worker_count": len(worker_counts),
        "duplicate_url_count": duplicate_urls,
        "duplicate_sample_id_count": duplicate_ids,
        "inspected_url_count": inspect_count,
        "source_check_enabled": source_check_enabled,
        "source_exists_count": source_exists_count,
        "source_exists_fraction": _fraction(source_exists_count, inspect_count),
        "raw_present_count": len(raw_present_ids),
        "feature_present_count": len(feature_present_ids),
        "cached_both_count": len(cached_both_ids),
        "raw_present_fraction": _fraction(len(raw_present_ids), inspect_count),
        "feature_present_fraction": _fraction(len(feature_present_ids), inspect_count),
        "cached_both_fraction": _fraction(len(cached_both_ids), inspect_count),
        "missing_raw_sample_ids": [sample_id for sample_id in inspected_ids if sample_id not in raw_by_id][:25],
        "missing_feature_sample_ids": [sample_id for sample_id in inspected_ids if sample_id not in feature_by_id][:25],
        "worker_clip_count_summary": _numeric_summary(worker_clip_counts),
        "feature_window_summary": window_summary,
        "all_sample_ids": sample_ids,
        "inspected_sample_ids": inspected_ids,
    }


def _feature_window_summary(paths: list[Path], *, max_files: int | None) -> dict[str, Any]:
    selected = paths[:max_files] if max_files is not None else paths
    counts: list[int] = []
    dims: Counter[int] = Counter()
    failed = 0
    progress_every = max(1, len(selected) // 10) if selected else 1
    for index, path in enumerate(selected, start=1):
        try:
            with np.load(path) as data:
                values = np.asarray(data["window_features"])
            if values.ndim == 2:
                counts.append(int(values.shape[0]))
                dims[int(values.shape[1])] += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        log_progress(
            "support_coverage_audit",
            "feature_window_stats_progress",
            index=index,
            total=len(selected),
            every=progress_every,
            failed=failed,
        )
    summary = _numeric_summary(counts)
    estimated_total = float(summary["mean"] * len(paths)) if counts else 0.0
    return {
        "available_feature_files": len(paths),
        "inspected_feature_files": len(selected),
        "failed_feature_files": failed,
        "window_count": summary,
        "feature_dim_counts": {str(key): int(value) for key, value in sorted(dims.items())},
        "estimated_total_feature_windows": estimated_total,
    }


def _expected_summary(expected: dict[str, Any] | None, split_reports: dict[str, Any]) -> dict[str, Any]:
    expected = expected or {}
    pretrain = split_reports.get("pretrain", {})
    new = split_reports.get("new", {})
    old_windows = float(expected.get("old_three_min_windows", 0.0) or 0.0)
    old_workers = float(expected.get("old_workers", 0.0) or 0.0)
    new_workers = float(expected.get("new_workers", 0.0) or 0.0)
    pretrain_cached = float(pretrain.get("cached_both_count", 0.0) or 0.0)
    pretrain_workers = float(pretrain.get("manifest_worker_count", 0.0) or 0.0)
    new_cached = float(new.get("cached_both_count", 0.0) or 0.0)
    return {
        "old_workers": old_workers,
        "new_workers": new_workers,
        "old_three_min_windows": old_windows,
        "pretrain_manifest_worker_fraction_of_expected": _fraction(pretrain_workers, old_workers),
        "pretrain_cached_clip_fraction_of_expected_old_windows": _fraction(pretrain_cached, old_windows),
        "new_cached_clip_fraction_of_expected_new_workers": _fraction(new_cached, new_workers),
    }


def _files_by_stem(root: Path, glob_pattern: str) -> dict[str, Path]:
    return {path.stem: path for path in root.glob(glob_pattern)}


def _numeric_summary(values: Iterable[float | int]) -> dict[str, float]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": float(len(numbers)),
        "mean": float(sum(numbers) / len(numbers)),
        "min": numbers[0],
        "p50": _percentile(numbers, 50.0),
        "p90": _percentile(numbers, 90.0),
        "max": numbers[-1],
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _drop_internal_ids(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"all_sample_ids", "inspected_sample_ids"}
    }


def _loggable_split_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_url_count": report["manifest_url_count"],
        "manifest_worker_count": report["manifest_worker_count"],
        "inspected_url_count": report["inspected_url_count"],
        "cached_both_count": report["cached_both_count"],
        "source_exists_count": report["source_exists_count"],
    }


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Support coverage config must include a '{key}' object.")
    return value
