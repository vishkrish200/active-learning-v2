from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.window_features import window_feature_matrix_from_jsonl


def build_new_split_cache(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    source_root = Path(config["source"]["root"])
    target_root = Path(config["target"]["root"])
    source_split = str(config["source"].get("split", "val"))
    source_manifest = target_root / str(config["target"]["source_manifest"])
    new_manifest = target_root / str(config["target"]["new_manifest"])
    empty_val_manifest = target_root / str(config["target"]["empty_val_manifest"])
    feature_dir = target_root / str(config["target"].get("feature_dir", "cache/features"))
    raw_dir = target_root / str(config["target"].get("raw_dir", "cache/raw"))
    limit = int(config["execution"].get("smoke_samples", 8)) if smoke else config["execution"].get("full_samples")
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("cache_new_split sample limit must be positive.")

    urls = read_manifest_urls(source_manifest)
    selected_urls = urls[:limit] if limit is not None else urls
    feature_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    new_manifest.parent.mkdir(parents=True, exist_ok=True)
    empty_val_manifest.parent.mkdir(parents=True, exist_ok=True)

    log_event(
        "cache_new_split",
        "start",
        smoke=smoke,
        n_manifest_urls=len(urls),
        n_selected=len(selected_urls),
        source_split=source_split,
    )
    written = 0
    skipped = 0
    missing_source = 0
    progress_every = max(1, len(selected_urls) // 10) if selected_urls else 1
    for index, url in enumerate(selected_urls, start=1):
        sample_id = hash_manifest_url(url)
        source_path = source_root / source_split / Path(url).name
        raw_path = raw_dir / f"{sample_id}.jsonl"
        feature_path = feature_dir / f"{sample_id}.npz"
        if not source_path.exists():
            missing_source += 1
            log_event("cache_new_split", "missing_source", url=url, source_path=str(source_path))
            continue
        if raw_path.exists() and feature_path.exists():
            skipped += 1
        else:
            shutil.copyfile(source_path, raw_path)
            window_features, clip_features = window_feature_matrix_from_jsonl(raw_path)
            np.savez(feature_path, window_features=window_features, clip_features=clip_features)
            written += 1
        log_progress(
            "cache_new_split",
            "progress",
            index=index,
            total=len(selected_urls),
            every=progress_every,
            written=written,
            skipped=skipped,
            missing_source=missing_source,
        )

    if not smoke:
        new_manifest.write_text("\n".join(urls) + "\n", encoding="utf-8")
        empty_val_manifest.write_text("", encoding="utf-8")
    else:
        new_manifest.with_suffix(".smoke.txt").write_text("\n".join(selected_urls) + "\n", encoding="utf-8")

    report = {
        "mode": "smoke" if smoke else "full",
        "source_split": source_split,
        "n_manifest_urls": len(urls),
        "n_selected": len(selected_urls),
        "written": written,
        "skipped": skipped,
        "missing_source": missing_source,
        "new_manifest": str(new_manifest),
        "empty_val_manifest": str(empty_val_manifest),
    }
    report_path = target_root / str(config["target"].get("report_path", "cache/manifests/new_cache_report.json"))
    if smoke:
        report_path = report_path.with_name("new_cache_report_smoke.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log_event("cache_new_split", "done", **report)
    return report
