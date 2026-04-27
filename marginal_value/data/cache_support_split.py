from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.window_features import window_feature_matrix_from_jsonl


def build_support_split_cache(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    source_root = Path(config["source"]["root"])
    target_root = Path(config["target"]["root"])
    source_split = str(config["source"].get("split", "pretrain"))
    source_manifest = target_root / str(config["target"]["source_manifest"])
    feature_dir = target_root / str(config["target"].get("feature_dir", "cache/features"))
    raw_dir = target_root / str(config["target"].get("raw_dir", "cache/raw"))
    report_path = target_root / str(config["target"].get("report_path", f"cache/manifests/{source_split}_cache_report.json"))

    urls = read_manifest_urls(source_manifest)
    selected_urls = _select_urls(urls, config=config, source_root=source_root, source_split=source_split, smoke=smoke)
    shard = _shard_config(config)
    n_selected_before_shard = len(selected_urls)
    if shard["num_shards"] > 1 and not smoke:
        selected_urls = [
            url
            for index, url in enumerate(selected_urls)
            if index % shard["num_shards"] == shard["shard_index"]
        ]
    feature_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    log_event(
        "cache_support_split",
        "start",
        smoke=smoke,
        source_split=source_split,
        n_manifest_urls=len(urls),
        n_selected=len(selected_urls),
        n_selected_before_shard=n_selected_before_shard,
        selection_strategy=str(config.get("selection", {}).get("strategy", "first_n")),
        shard_index=shard["shard_index"],
        num_shards=shard["num_shards"],
    )
    written = 0
    skipped = 0
    missing_source = 0
    malformed_source = 0
    malformed_examples: list[dict[str, str]] = []
    raw_copied = 0
    feature_written = 0
    configured_progress_every = config.get("execution", {}).get("progress_every")
    progress_every = int(configured_progress_every) if configured_progress_every is not None else max(1, len(selected_urls) // 20) if selected_urls else 1
    if progress_every <= 0:
        raise ValueError("execution.progress_every must be positive when provided.")
    for index, url in enumerate(selected_urls, start=1):
        sample_id = hash_manifest_url(url)
        source_path = _source_path_for_url(source_root, split=source_split, url=url)
        raw_path = raw_dir / f"{sample_id}.jsonl"
        feature_path = feature_dir / f"{sample_id}.npz"
        if index == 1:
            log_event(
                "cache_support_split",
                "first_selected_item",
                sample_id=sample_id,
                source_path=str(source_path),
                raw_path=str(raw_path),
                feature_path=str(feature_path),
            )
        if not source_path.exists():
            missing_source += 1
            log_event("cache_support_split", "missing_source", url=url, source_path=str(source_path))
            continue
        if raw_path.exists() and feature_path.exists():
            skipped += 1
        else:
            if not raw_path.exists():
                shutil.copyfile(source_path, raw_path)
                raw_copied += 1
            if not feature_path.exists():
                try:
                    window_features, clip_features = window_feature_matrix_from_jsonl(raw_path)
                except ValueError as exc:
                    malformed_source += 1
                    if len(malformed_examples) < 20:
                        malformed_examples.append(
                            {
                                "url": url,
                                "source_path": str(source_path),
                                "raw_path": str(raw_path),
                                "error": str(exc),
                            }
                        )
                    log_event(
                        "cache_support_split",
                        "malformed_source",
                        url=url,
                        source_path=str(source_path),
                        raw_path=str(raw_path),
                        error=str(exc),
                    )
                    continue
                np.savez(feature_path, window_features=window_features, clip_features=clip_features)
                feature_written += 1
            written += 1
        log_progress(
            "cache_support_split",
            "progress",
            index=index,
            total=len(selected_urls),
            every=progress_every,
            written=written,
            skipped=skipped,
            missing_source=missing_source,
            malformed_source=malformed_source,
            raw_copied=raw_copied,
            feature_written=feature_written,
        )

    report = {
        "mode": "smoke" if smoke else "full",
        "source_split": source_split,
        "n_manifest_urls": len(urls),
        "n_selected": len(selected_urls),
        "n_selected_before_shard": n_selected_before_shard,
        "written": written,
        "skipped": skipped,
        "missing_source": missing_source,
        "malformed_source": malformed_source,
        "malformed_examples": malformed_examples,
        "raw_copied": raw_copied,
        "feature_written": feature_written,
        "selection": _selection_report(selected_urls, config=config),
        "shard": shard,
        "report_path": str(report_path),
    }
    output_report_path = _report_path_for_mode(report_path, smoke=smoke, shard=shard)
    report["report_path"] = str(output_report_path)
    output_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log_event("cache_support_split", "done", **report)
    return report


def _shard_config(config: dict[str, Any]) -> dict[str, int]:
    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        execution = {}
    num_shards = int(execution.get("num_shards", 1))
    shard_index = int(execution.get("shard_index", 0))
    if num_shards <= 0:
        raise ValueError("execution.num_shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("execution.shard_index must satisfy 0 <= shard_index < num_shards.")
    return {"num_shards": num_shards, "shard_index": shard_index}


def _report_path_for_mode(report_path: Path, *, smoke: bool, shard: dict[str, int]) -> Path:
    suffix_parts: list[str] = []
    if smoke:
        suffix_parts.append("smoke")
    if shard["num_shards"] > 1:
        suffix_parts.append(f"shard{shard['shard_index']:02d}of{shard['num_shards']:02d}")
    if not suffix_parts:
        return report_path
    return report_path.with_name(f"{report_path.stem}_{'_'.join(suffix_parts)}{report_path.suffix}")


def _select_urls(
    urls: list[str],
    *,
    config: dict[str, Any],
    source_root: Path,
    source_split: str,
    smoke: bool,
) -> list[str]:
    selection = config.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    strategy = str(selection.get("strategy", "first_n"))
    smoke_limit = int(config["execution"].get("smoke_samples", 8))
    full_limit = config["execution"].get("full_samples")
    selection_limit = smoke_limit if smoke else int(full_limit) if full_limit is not None else None
    if strategy == "first_n":
        selected = list(urls)
    elif strategy == "worker_coverage":
        require_source_exists = bool(selection.get("require_source_exists", True))
        selected = _worker_coverage_urls(
            urls,
            clips_per_worker=int(selection.get("clips_per_worker", 1)),
            max_workers=selection.get("max_workers"),
            source_root=source_root,
            source_split=source_split,
            require_source_exists=require_source_exists,
            max_selected=selection_limit,
        )
    elif strategy == "all":
        selected = list(urls)
    elif strategy == "source_existing_all":
        selected = _source_existing_urls(
            urls,
            source_root=source_root,
            source_split=source_split,
            max_selected=selection_limit,
        )
    else:
        raise ValueError("selection.strategy must be 'first_n', 'worker_coverage', 'all', or 'source_existing_all'.")

    if not smoke and full_limit is not None:
        selected = selected[: int(full_limit)]
    if smoke:
        selected = selected[:smoke_limit]
    return selected


def _source_existing_urls(
    urls: list[str],
    *,
    source_root: Path,
    source_split: str,
    max_selected: int | None,
) -> list[str]:
    selected: list[str] = []
    progress_every = max(1, len(urls) // 20) if urls else 1
    for index, url in enumerate(urls, start=1):
        if _source_path_for_url(source_root, split=source_split, url=url).exists():
            selected.append(url)
        log_progress(
            "cache_support_split",
            "source_existing_selection_progress",
            index=index,
            total=len(urls),
            every=progress_every,
            selected=len(selected),
        )
        if max_selected is not None and len(selected) >= max_selected:
            break
    return selected


def _worker_coverage_urls(
    urls: list[str],
    *,
    clips_per_worker: int,
    max_workers: object | None,
    source_root: Path | None,
    source_split: str,
    require_source_exists: bool,
    max_selected: int | None,
) -> list[str]:
    if clips_per_worker <= 0:
        raise ValueError("selection.clips_per_worker must be positive.")
    worker_counts: dict[str, int] = defaultdict(int)
    selected: list[str] = []
    max_worker_count = int(max_workers) if max_workers is not None else None
    for url in urls:
        worker_id = _worker_id_from_url(url)
        if require_source_exists:
            if source_root is None:
                raise ValueError("source_root is required when require_source_exists is true.")
            if not _source_path_for_url(source_root, split=source_split, url=url).exists():
                continue
        if max_worker_count is not None and worker_id not in worker_counts and len(worker_counts) >= max_worker_count:
            continue
        if worker_counts[worker_id] >= clips_per_worker:
            continue
        worker_counts[worker_id] += 1
        selected.append(url)
        if max_selected is not None and len(selected) >= max_selected:
            break
    return selected


def _source_path_for_url(source_root: Path, *, split: str, url: str) -> Path:
    nested_path = _nested_source_path_for_url(source_root, split=split, url=url)
    flat_path = source_root / _flat_source_name_for_url(url)
    if nested_path.exists():
        return nested_path
    if flat_path.exists():
        return flat_path
    return nested_path


def _nested_source_path_for_url(source_root: Path, *, split: str, url: str) -> Path:
    parsed = urlparse(url)
    parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
    if split in parts:
        split_index = parts.index(split)
        remainder = parts[split_index + 1 :]
        if remainder:
            return source_root / split / Path(*remainder)
    return source_root / split / Path(parsed.path).name


def _flat_source_name_for_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [parsed.netloc, *[part for part in PurePosixPath(parsed.path).parts if part and part != "/"]]
    return "__".join(parts)


def _worker_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
    for part in parts:
        if part.startswith("worker"):
            return part
    return hash_manifest_url(url)


def _selection_report(selected_urls: list[str], *, config: dict[str, Any]) -> dict[str, object]:
    workers = {_worker_id_from_url(url) for url in selected_urls}
    return {
        "strategy": str(config.get("selection", {}).get("strategy", "first_n")),
        "n_urls": len(selected_urls),
        "n_workers": len(workers),
        "clips_per_worker": config.get("selection", {}).get("clips_per_worker"),
        "max_workers": config.get("selection", {}).get("max_workers"),
    }
