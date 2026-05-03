from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.window_features import window_feature_matrix_from_jsonl


def load_manifest_url_cache_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_manifest_url_cache_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    target = _required_mapping(config, "target")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal":
        raise ValueError("Manifest URL cache must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(target.get("root", "")).startswith("/data"):
        raise ValueError("target.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    for key in ("source_manifest", "cached_manifest"):
        if not str(target.get(key, "")).startswith("cache/manifests/"):
            raise ValueError(f"target.{key} must be under cache/manifests/.")
    if int(execution.get("smoke_samples", 8)) <= 0:
        raise ValueError("execution.smoke_samples must be positive.")
    if int(execution.get("progress_every", 1000)) <= 0:
        raise ValueError("execution.progress_every must be positive.")
    if int(execution.get("workers", 1)) <= 0:
        raise ValueError("execution.workers must be positive.")
    if int(execution.get("max_pending", 1)) <= 0:
        raise ValueError("execution.max_pending must be positive.")
    if float(execution.get("download_timeout_seconds", 60.0)) <= 0.0:
        raise ValueError("execution.download_timeout_seconds must be positive.")


def cache_manifest_urls(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_manifest_url_cache_config(config)
    mode = "smoke" if smoke else "full"
    target_root = Path(config["target"]["root"])
    raw_dir = target_root / str(config["target"].get("raw_dir", "cache/raw"))
    feature_dir = target_root / str(config["target"].get("feature_dir", "cache/features"))
    source_manifest_path = target_root / str(config["target"]["source_manifest"])
    cached_manifest_path = target_root / str(config["target"]["cached_manifest"])
    output_dir = Path(config["artifacts"]["output_dir"])
    report_path = output_dir / f"manifest_url_cache_{mode}.json"
    if smoke:
        cached_manifest_path = cached_manifest_path.with_suffix(".smoke.txt")

    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    cached_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = read_manifest_urls(source_manifest_path)
    selection = config.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    strategy = str(selection.get("strategy", "missing_cache"))
    if strategy != "missing_cache":
        raise ValueError("selection.strategy must be 'missing_cache'.")
    source_manifest_url_count = len(urls)
    manifest_start_index, manifest_end_index = _resolve_manifest_range(selection, source_manifest_url_count)
    urls = urls[manifest_start_index:manifest_end_index]
    smoke_limit = int(config["execution"].get("smoke_samples", 8))
    full_limit = config["execution"].get("full_samples")
    selected_limit = smoke_limit if smoke else int(full_limit) if full_limit is not None else None
    progress_every = int(config["execution"].get("progress_every", 1000))
    range_is_sharded = manifest_start_index != 0 or manifest_end_index != source_manifest_url_count
    if selected_limit is not None:
        progress_total = selected_limit
    elif range_is_sharded:
        progress_total = len(urls)
    else:
        progress_total = int(config["execution"].get("full_selected_progress_hint", len(urls)))
    workers = int(config["execution"].get("workers", 1))
    max_pending = int(config["execution"].get("max_pending", max(4, workers * 4)))
    download_timeout = float(config["execution"].get("download_timeout_seconds", 60.0))
    shard_id = str(config["execution"].get("shard_id", "")).strip()
    if shard_id:
        report_path = _suffixed_path(report_path, _safe_suffix(shard_id))
        cached_manifest_path = _suffixed_path(cached_manifest_path, _safe_suffix(shard_id))

    log_event(
        "cache_manifest_urls",
        "start",
        mode=mode,
        source_manifest_url_count=source_manifest_url_count,
        manifest_start_index=manifest_start_index,
        manifest_end_index=manifest_end_index,
        manifest_url_count=len(urls),
        strategy=strategy,
        selected_limit=selected_limit,
        workers=workers,
        shard_id=shard_id,
    )

    selected = 0
    skipped_existing = 0
    submitted = 0
    downloaded = 0
    feature_written = 0
    raw_already_present = 0
    raw_redownloaded_after_feature_error = 0
    failed = 0
    failure_examples: list[dict[str, str]] = []
    pending: list[Future[dict[str, Any]]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for url in urls:
            sample_id = hash_manifest_url(url)
            raw_path = raw_dir / f"{sample_id}.jsonl"
            feature_path = feature_dir / f"{sample_id}.npz"
            if raw_path.exists() and feature_path.exists():
                skipped_existing += 1
                continue
            selected += 1
            pending.append(
                executor.submit(
                    _cache_one_url,
                    url,
                    str(raw_path),
                    str(feature_path),
                    download_timeout,
                )
            )
            submitted += 1
            if len(pending) >= max_pending:
                result = _collect_one(pending.pop(0), failure_examples)
                downloaded += int(result.get("downloaded", 0))
                feature_written += int(result.get("feature_written", 0))
                raw_already_present += int(result.get("raw_already_present", 0))
                raw_redownloaded_after_feature_error += int(result.get("raw_redownloaded_after_feature_error", 0))
                failed += int(result.get("failed", 0))
            log_progress(
                "cache_manifest_urls",
                "progress",
                index=selected,
                total=progress_total,
                every=progress_every,
                mode=mode,
                skipped_existing=skipped_existing,
                submitted=submitted,
                downloaded=downloaded,
                feature_written=feature_written,
                failed=failed,
            )
            if selected_limit is not None and selected >= selected_limit:
                break
        for future in pending:
            result = _collect_one(future, failure_examples)
            downloaded += int(result.get("downloaded", 0))
            feature_written += int(result.get("feature_written", 0))
            raw_already_present += int(result.get("raw_already_present", 0))
            raw_redownloaded_after_feature_error += int(result.get("raw_redownloaded_after_feature_error", 0))
            failed += int(result.get("failed", 0))

    cached_urls = [
        url
        for url in urls
        if _cache_paths_present(url, raw_dir=raw_dir, feature_dir=feature_dir)
    ]
    cached_manifest_path.write_text("".join(f"{url}\n" for url in cached_urls), encoding="utf-8")
    report = {
        "mode": mode,
        "manifest": str(source_manifest_path),
        "source_manifest_url_count": source_manifest_url_count,
        "manifest_start_index": manifest_start_index,
        "manifest_end_index": manifest_end_index,
        "manifest_url_count": len(urls),
        "shard_id": shard_id,
        "selection_strategy": strategy,
        "selected": selected,
        "selected_limit": selected_limit,
        "skipped_existing": skipped_existing,
        "submitted": submitted,
        "downloaded": downloaded,
        "raw_already_present": raw_already_present,
        "raw_redownloaded_after_feature_error": raw_redownloaded_after_feature_error,
        "feature_written": feature_written,
        "failed": failed,
        "failure_examples": failure_examples,
        "cached_manifest": str(cached_manifest_path),
        "cached_both_count": len(cached_urls),
        "raw_dir": str(raw_dir),
        "feature_dir": str(feature_dir),
        "report_path": str(report_path),
        "notes": [
            "This job downloads directly from manifest URLs when raw/features are missing.",
            "cached_manifest contains source manifest URLs that have both raw JSONL and feature NPZ after the run.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log_event("cache_manifest_urls", "done", **report)
    if not smoke and bool(config["execution"].get("fail_if_incomplete", False)):
        if failed or len(cached_urls) != len(urls):
            raise RuntimeError(
                "Manifest URL cache incomplete: "
                f"{len(cached_urls)}/{len(urls)} URLs have raw JSONL and feature NPZ; "
                f"{failed} downloads or feature extractions failed. "
                f"Report written to {report_path}."
            )
    return report


def _cache_one_url(url: str, raw_path_text: str, feature_path_text: str, download_timeout: float) -> dict[str, Any]:
    raw_path = Path(raw_path_text)
    feature_path = Path(feature_path_text)
    result: dict[str, Any] = {
        "url": url,
        "downloaded": 0,
        "raw_already_present": 0,
        "feature_written": 0,
        "raw_redownloaded_after_feature_error": 0,
        "failed": 0,
    }
    try:
        if raw_path.exists():
            result["raw_already_present"] = 1
        else:
            _download_url(url, raw_path, timeout=download_timeout)
            result["downloaded"] = 1
        if not feature_path.exists():
            try:
                window_features, clip_features = window_feature_matrix_from_jsonl(raw_path)
            except ValueError:
                if not result["raw_already_present"]:
                    raise
                _download_url(url, raw_path, timeout=download_timeout)
                result["raw_redownloaded_after_feature_error"] = 1
                window_features, clip_features = window_feature_matrix_from_jsonl(raw_path)
            np.savez(feature_path, window_features=window_features, clip_features=clip_features)
            result["feature_written"] = 1
        return result
    except Exception as exc:  # noqa: BLE001 - report and continue the batch.
        result["failed"] = 1
        result["error"] = str(exc)
        try:
            if raw_path.exists() and not feature_path.exists() and result.get("downloaded"):
                raw_path.unlink()
        except OSError:
            pass
        return result


def _download_url(url: str, output_path: Path, *, timeout: float) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{threading.get_ident()}.{uuid.uuid4().hex}")
    request = Request(url, headers={"User-Agent": "active-learning-v2-cache/1.0"})
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if int(status) >= 400:
            raise RuntimeError(f"download failed with HTTP {status}")
        with tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)


def _collect_one(future: Future[dict[str, Any]], failure_examples: list[dict[str, str]]) -> dict[str, Any]:
    result = future.result()
    if result.get("failed") and len(failure_examples) < 20:
        failure_examples.append(
            {
                "url": str(result.get("url", "")),
                "error": str(result.get("error", "")),
            }
        )
    return result


def _cache_paths_present(url: str, *, raw_dir: Path, feature_dir: Path) -> bool:
    sample_id = hash_manifest_url(url)
    return (raw_dir / f"{sample_id}.jsonl").exists() and (feature_dir / f"{sample_id}.npz").exists()


def _resolve_manifest_range(selection: dict[str, Any], url_count: int) -> tuple[int, int]:
    start = int(selection.get("manifest_start_index", 0))
    end_value = selection.get("manifest_end_index", url_count)
    end = url_count if end_value is None else int(end_value)
    if start < 0:
        raise ValueError("selection.manifest_start_index must be non-negative.")
    if end < start:
        raise ValueError("selection.manifest_end_index must be greater than or equal to manifest_start_index.")
    if end > url_count:
        raise ValueError("selection.manifest_end_index must not exceed the manifest URL count.")
    return start, end


def _safe_suffix(value: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if not suffix:
        raise ValueError("execution.shard_id must contain at least one alphanumeric character when provided.")
    return suffix


def _suffixed_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}.{suffix}{path.suffix}")


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing config mapping: {key}")
    return value
