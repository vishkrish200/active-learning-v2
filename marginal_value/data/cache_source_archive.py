from __future__ import annotations

import json
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from marginal_value.data.source_archive_audit import _open_tar_archive, _url_from_archive_member, _worker_id_from_url
from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.window_features import window_feature_matrix_from_jsonl


def load_source_archive_cache_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_source_archive_cache_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    source = _required_mapping(config, "source")
    target = _required_mapping(config, "target")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal":
        raise ValueError("Source archive cache must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(source.get("root", "")).startswith("/source"):
        raise ValueError("source.root must be mounted under /source.")
    if not allow_local_paths and not str(source.get("archive_path", "")).startswith("/source"):
        raise ValueError("source.archive_path must be mounted under /source.")
    if not allow_local_paths and not str(target.get("root", "")).startswith("/data"):
        raise ValueError("target.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    for key in ("source_manifest", "extracted_manifest", "union_manifest", "archive_cached_manifest"):
        if not str(target.get(key, "")).startswith("cache/manifests/"):
            raise ValueError(f"target.{key} must be under cache/manifests/.")
    if int(execution.get("smoke_samples", 8)) <= 0:
        raise ValueError("execution.smoke_samples must be positive.")
    if int(execution.get("progress_every", 1000)) <= 0:
        raise ValueError("execution.progress_every must be positive.")
    if int(execution.get("full_selected_progress_hint", 40000)) <= 0:
        raise ValueError("execution.full_selected_progress_hint must be positive.")
    if int(execution.get("feature_workers", 1)) <= 0:
        raise ValueError("execution.feature_workers must be positive.")
    if not str(source.get("split", "")):
        raise ValueError("source.split must be non-empty.")
    if not str(source.get("url_prefix", "")):
        raise ValueError("source.url_prefix must be non-empty.")


def cache_source_archive(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_source_archive_cache_config(config)
    mode = "smoke" if smoke else "full"
    source_root = Path(config["source"]["root"])
    archive_path = Path(config["source"]["archive_path"])
    target_root = Path(config["target"]["root"])
    raw_dir = target_root / str(config["target"].get("raw_dir", "cache/raw"))
    feature_dir = target_root / str(config["target"].get("feature_dir", "cache/features"))
    manifest_path = target_root / str(config["target"]["source_manifest"])
    extracted_manifest_path = target_root / str(config["target"]["extracted_manifest"])
    union_manifest_path = target_root / str(config["target"]["union_manifest"])
    archive_cached_manifest_path = target_root / str(config["target"]["archive_cached_manifest"])
    output_dir = Path(config["artifacts"]["output_dir"])
    report_path = output_dir / f"source_archive_cache_{mode}.json"
    if smoke:
        union_manifest_path = union_manifest_path.with_suffix(".smoke.txt")
        archive_cached_manifest_path = archive_cached_manifest_path.with_suffix(".smoke.txt")

    raw_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    union_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archive_cached_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_urls = read_manifest_urls(manifest_path)
    manifest_url_set = set(manifest_urls)
    extracted_urls = set(read_manifest_urls(extracted_manifest_path))
    strategy = str(config.get("selection", {}).get("strategy", "missing_from_extracted"))
    smoke_limit = int(config["execution"].get("smoke_samples", 8))
    selected_limit = smoke_limit if smoke else config["execution"].get("full_samples")
    selected_limit = int(selected_limit) if selected_limit is not None else None
    progress_every = int(config["execution"].get("progress_every", 1000))
    progress_total = selected_limit or int(config["execution"].get("full_selected_progress_hint", 40000))
    feature_workers = int(config["execution"].get("feature_workers", 1))
    max_pending_features = int(config["execution"].get("max_pending_features", max(4, feature_workers * 4)))

    log_event(
        "cache_source_archive",
        "start",
        mode=mode,
        archive_path=str(archive_path),
        manifest_url_count=len(manifest_urls),
        extracted_url_count=len(extracted_urls),
        strategy=strategy,
        selected_limit=selected_limit,
        feature_workers=feature_workers,
    )

    archive_data_members = 0
    selected = 0
    skipped_existing = 0
    raw_copied = 0
    feature_written = 0
    feature_skipped_existing = 0
    metadata_members = 0
    skipped_members = 0
    not_in_manifest = 0
    not_selected_by_strategy = 0
    extract_failed = 0
    malformed_source = 0
    malformed_examples: list[dict[str, str]] = []
    cached_archive_urls: list[str] = []
    pending: list[Future[tuple[str, str, str]]] = []

    with ThreadPoolExecutor(max_workers=feature_workers) as executor:
        with _open_tar_archive(archive_path) as archive:
            for member in archive:
                if not member.isfile():
                    skipped_members += 1
                    continue
                url = _url_from_archive_member(
                    member.name,
                    split=str(config["source"]["split"]),
                    url_prefix=str(config["source"]["url_prefix"]),
                )
                if url == "metadata":
                    metadata_members += 1
                    continue
                if not url:
                    skipped_members += 1
                    continue
                archive_data_members += 1
                if url not in manifest_url_set:
                    not_in_manifest += 1
                    continue
                if not _selected_by_strategy(url, strategy=strategy, extracted_urls=extracted_urls):
                    not_selected_by_strategy += 1
                    continue
                selected += 1
                sample_id = hash_manifest_url(url)
                raw_path = raw_dir / f"{sample_id}.jsonl"
                feature_path = feature_dir / f"{sample_id}.npz"
                if raw_path.exists() and feature_path.exists():
                    skipped_existing += 1
                    cached_archive_urls.append(url)
                else:
                    if not raw_path.exists():
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            extract_failed += 1
                            if len(malformed_examples) < 20:
                                malformed_examples.append({"url": url, "member": member.name, "error": "extractfile returned None"})
                            continue
                        with raw_path.open("wb") as handle:
                            shutil.copyfileobj(extracted, handle)
                        raw_copied += 1
                    if feature_path.exists():
                        feature_skipped_existing += 1
                        cached_archive_urls.append(url)
                    else:
                        pending.append(executor.submit(_write_feature_file, url, str(raw_path), str(feature_path)))
                        if len(pending) >= max_pending_features:
                            done = pending.pop(0)
                            if _collect_feature_result(done, malformed_examples):
                                feature_written += 1
                                cached_archive_urls.append(done.result()[0])
                            else:
                                malformed_source += 1
                if selected_limit is not None and selected >= selected_limit:
                    break
                log_progress(
                    "cache_source_archive",
                    "progress",
                    index=selected,
                    total=progress_total,
                    every=progress_every,
                    mode=mode,
                    archive_data_members=archive_data_members,
                    skipped_existing=skipped_existing,
                    raw_copied=raw_copied,
                    feature_written=feature_written,
                    malformed_source=malformed_source,
                )

        for done in pending:
            if _collect_feature_result(done, malformed_examples):
                feature_written += 1
                cached_archive_urls.append(done.result()[0])
            else:
                malformed_source += 1

    cached_archive_url_set = set(cached_archive_urls)
    union_urls = [url for url in manifest_urls if url in extracted_urls or url in cached_archive_url_set]
    archive_cached_manifest_path.write_text("".join(f"{url}\n" for url in cached_archive_urls), encoding="utf-8")
    union_manifest_path.write_text("".join(f"{url}\n" for url in union_urls), encoding="utf-8")

    worker_count = len({_worker_id_from_url(url) for url in cached_archive_url_set})
    union_worker_count = len({_worker_id_from_url(url) for url in union_urls})
    report = {
        "mode": mode,
        "source_root": str(source_root),
        "archive_path": str(archive_path),
        "manifest_url_count": len(manifest_urls),
        "extracted_url_count": len(extracted_urls),
        "archive_data_members_seen": archive_data_members,
        "selected": selected,
        "selected_limit": selected_limit,
        "selection_strategy": strategy,
        "cached_archive_url_count": len(cached_archive_url_set),
        "cached_archive_worker_count": worker_count,
        "union_url_count": len(union_urls),
        "union_worker_count": union_worker_count,
        "skipped_existing": skipped_existing,
        "raw_copied": raw_copied,
        "feature_written": feature_written,
        "feature_skipped_existing": feature_skipped_existing,
        "metadata_members": metadata_members,
        "skipped_members": skipped_members,
        "not_in_manifest": not_in_manifest,
        "not_selected_by_strategy": not_selected_by_strategy,
        "extract_failed": extract_failed,
        "malformed_source": malformed_source,
        "malformed_examples": malformed_examples,
        "archive_cached_manifest": str(archive_cached_manifest_path),
        "union_manifest": str(union_manifest_path),
        "report_path": str(report_path),
        "notes": [
            "This job caches archive members directly into target cache/raw and cache/features.",
            "The union manifest contains extracted physical-source URLs plus successfully cached archive URLs, ordered by the original source manifest.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log_event("cache_source_archive", "done", **report)
    return report


def _selected_by_strategy(url: str, *, strategy: str, extracted_urls: set[str]) -> bool:
    if strategy == "missing_from_extracted":
        return url not in extracted_urls
    if strategy == "archive_manifest_all":
        return True
    raise ValueError("selection.strategy must be 'missing_from_extracted' or 'archive_manifest_all'.")


def _write_feature_file(url: str, raw_path: str, feature_path: str) -> tuple[str, str, str]:
    window_features, clip_features = window_feature_matrix_from_jsonl(raw_path)
    np.savez(feature_path, window_features=window_features, clip_features=clip_features)
    return url, raw_path, feature_path


def _collect_feature_result(future: Future[tuple[str, str, str]], malformed_examples: list[dict[str, str]]) -> bool:
    try:
        future.result()
        return True
    except ValueError as exc:
        if len(malformed_examples) < 20:
            malformed_examples.append({"url": "unknown", "member": "", "error": str(exc)})
        return False


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing config mapping: {key}")
    return value
