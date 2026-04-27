from __future__ import annotations

import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from marginal_value.data.cache_support_split import _source_path_for_url, _worker_id_from_url
from marginal_value.data.split_manifest import read_manifest_urls
from marginal_value.logging_utils import log_event, log_progress


def load_source_inventory_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_source_inventory_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    source = _required_mapping(config, "source")
    target = _required_mapping(config, "target")
    artifacts = _required_mapping(config, "artifacts")
    manifests = _required_mapping(target, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Source inventory must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(source.get("root", "")).startswith("/source"):
        raise ValueError("Source inventory source.root must be mounted under /source.")
    if not allow_local_paths and not str(target.get("root", "")).startswith("/data"):
        raise ValueError("Source inventory target.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("Source inventory artifacts.output_dir must be mounted under /artifacts.")
    if not manifests:
        raise ValueError("Source inventory target.manifests must not be empty.")
    for split, manifest in manifests.items():
        if not str(split):
            raise ValueError("Source inventory manifest split names must be non-empty.")
        if not str(manifest).startswith("cache/manifests/"):
            raise ValueError("Source inventory manifest paths must be under cache/manifests/.")
    physical_manifest = target.get("physical_source_manifest")
    if physical_manifest is not None and not str(physical_manifest).startswith("cache/manifests/"):
        raise ValueError("target.physical_source_manifest must be under cache/manifests/ when provided.")
    if "smoke_manifest_samples" in execution and int(execution["smoke_manifest_samples"]) <= 0:
        raise ValueError("execution.smoke_manifest_samples must be positive when provided.")
    if "smoke_scan_files" in execution and int(execution["smoke_scan_files"]) <= 0:
        raise ValueError("execution.smoke_scan_files must be positive when provided.")


def run_source_inventory(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_source_inventory_config(config)
    mode = "smoke" if smoke else "full"
    source_root = Path(config["source"]["root"])
    target_root = Path(config["target"]["root"])
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_by_manifest = config.get("source", {}).get("split_by_manifest", {})
    if split_by_manifest is None:
        split_by_manifest = {}
    if not isinstance(split_by_manifest, dict):
        raise ValueError("source.split_by_manifest must be an object when provided.")

    log_event("source_inventory", "start", mode=mode, source_root=str(source_root), target_root=str(target_root))
    source_report = _scan_source_root(
        source_root,
        scan_splits=[str(split) for split in config["source"].get("scan_splits", [])],
        smoke=smoke,
        smoke_scan_files=int(config["execution"].get("smoke_scan_files", 512)),
    )
    manifest_reports: dict[str, Any] = {}
    for split, manifest_relpath in sorted(config["target"]["manifests"].items()):
        manifest_path = target_root / str(manifest_relpath)
        urls = read_manifest_urls(manifest_path)
        source_split = str(split_by_manifest.get(split, split))
        manifest_reports[str(split)] = _audit_manifest_sources(
            split=str(split),
            source_split=source_split,
            urls=urls,
            source_root=source_root,
            smoke=smoke,
            smoke_samples=int(config["execution"].get("smoke_manifest_samples", 512)),
        )
    physical_manifest_report = _maybe_write_physical_source_manifest(
        config,
        source_root=source_root,
        target_root=target_root,
        smoke=smoke,
    )

    report = {
        "mode": mode,
        "source_root": str(source_root),
        "target_root": str(target_root),
        "source": source_report,
        "manifests": manifest_reports,
        "physical_source_manifest": physical_manifest_report,
        "notes": [
            "source_existing_count is manifest URL coverage in the mounted source mirror.",
            "source txt counts are physical files in the source volume, not cached ranking features.",
            "tar files are reported as archive candidates; this job does not extract them.",
            "physical_source_manifest, when configured, is a generated manifest over extracted txt files only.",
        ],
    }
    report_path = output_dir / f"source_inventory_{mode}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "report_path": str(report_path),
        "pretrain_manifest_url_count": manifest_reports.get("pretrain", {}).get("manifest_url_count", 0),
        "pretrain_source_existing_count": manifest_reports.get("pretrain", {}).get("source_existing_count", 0),
        "new_manifest_url_count": manifest_reports.get("new", {}).get("manifest_url_count", 0),
        "new_source_existing_count": manifest_reports.get("new", {}).get("source_existing_count", 0),
        "source_tar_count": len(source_report["tar_files"]),
        "physical_source_manifest_url_count": physical_manifest_report.get("url_count", 0),
    }
    log_event("source_inventory", "done", **result)
    return result


def _scan_source_root(
    source_root: Path,
    *,
    scan_splits: list[str],
    smoke: bool,
    smoke_scan_files: int,
) -> dict[str, Any]:
    split_reports = {
        split: _scan_split_tree(source_root / split, split=split, smoke=smoke, smoke_scan_files=smoke_scan_files)
        for split in scan_splits
    }
    flat_report = _scan_flat_files(source_root, smoke=smoke, smoke_scan_files=smoke_scan_files)
    tar_files = [
        {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(source_root.glob("*.tar*"))
        if path.is_file()
    ]
    return {
        "split_trees": split_reports,
        "flat_files": flat_report,
        "tar_files": tar_files,
    }


def _scan_split_tree(path: Path, *, split: str, smoke: bool, smoke_scan_files: int) -> dict[str, Any]:
    workers: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    examples: list[str] = []
    txt_count = 0
    inspected = 0
    iterator = sorted(file_path for file_path in path.rglob("*.txt") if _is_data_txt(file_path)) if path.exists() else []
    total_hint = len(iterator)
    for file_path in iterator:
        txt_count += 1
        inspected += 1
        suffixes[file_path.suffix] += 1
        worker_id = _worker_from_source_path(file_path, split=split)
        if worker_id:
            workers[worker_id] += 1
        if len(examples) < 10:
            examples.append(str(file_path))
        log_progress(
            "source_inventory",
            "split_tree_scan_progress",
            index=inspected,
            total=total_hint,
            every=max(1, total_hint // 10) if total_hint else 1,
            split=split,
            txt_count=txt_count,
        )
        if smoke and inspected >= smoke_scan_files:
            break
    return {
        "path": str(path),
        "exists": path.exists(),
        "txt_file_count": txt_count,
        "inspected_file_count": inspected,
        "worker_count": len(workers),
        "worker_clip_count_summary": _numeric_summary(workers.values()),
        "suffix_counts": dict(sorted(suffixes.items())),
        "examples": examples,
    }


def _scan_flat_files(source_root: Path, *, smoke: bool, smoke_scan_files: int) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    workers_by_split: dict[str, Counter[str]] = {}
    examples: list[str] = []
    inspected = 0
    files = sorted(path for path in source_root.glob("*.txt") if path.is_file() and _is_data_txt(path))
    total_hint = len(files)
    for file_path in files:
        inspected += 1
        parsed = _parse_flat_source_name(file_path.name)
        split = parsed.get("split", "unknown")
        split_counts[split] += 1
        worker_id = parsed.get("worker_id")
        if worker_id:
            workers_by_split.setdefault(split, Counter())[worker_id] += 1
        if len(examples) < 10:
            examples.append(str(file_path))
        log_progress(
            "source_inventory",
            "flat_scan_progress",
            index=inspected,
            total=total_hint,
            every=max(1, total_hint // 10) if total_hint else 1,
            txt_count=inspected,
        )
        if smoke and inspected >= smoke_scan_files:
            break
    return {
        "txt_file_count": inspected,
        "split_counts": dict(sorted(split_counts.items())),
        "worker_counts_by_split": {split: len(workers) for split, workers in sorted(workers_by_split.items())},
        "worker_clip_count_summary_by_split": {
            split: _numeric_summary(workers.values())
            for split, workers in sorted(workers_by_split.items())
        },
        "examples": examples,
    }


def _audit_manifest_sources(
    *,
    split: str,
    source_split: str,
    urls: list[str],
    source_root: Path,
    smoke: bool,
    smoke_samples: int,
) -> dict[str, Any]:
    inspect_count = min(len(urls), smoke_samples) if smoke else len(urls)
    inspected_urls = urls[:inspect_count]
    worker_counts = Counter(_worker_id_from_url(url) for url in urls)
    source_existing_count = 0
    nested_existing_count = 0
    flat_existing_count = 0
    missing_examples: list[str] = []
    progress_every = max(1, inspect_count // 20) if inspect_count else 1
    for index, url in enumerate(inspected_urls, start=1):
        source_path = _source_path_for_url(source_root, split=source_split, url=url)
        if source_path.exists():
            source_existing_count += 1
            if source_path.parent.name == source_split or source_split in source_path.parts:
                nested_existing_count += 1
            else:
                flat_existing_count += 1
        elif len(missing_examples) < 10:
            missing_examples.append(url)
        log_progress(
            "source_inventory",
            "manifest_source_progress",
            index=index,
            total=inspect_count,
            every=progress_every,
            split=split,
            source_existing_count=source_existing_count,
        )
    return {
        "split": split,
        "source_split": source_split,
        "manifest_url_count": len(urls),
        "manifest_worker_count": len(worker_counts),
        "inspected_url_count": inspect_count,
        "source_existing_count": source_existing_count,
        "source_existing_fraction": _fraction(source_existing_count, inspect_count),
        "nested_existing_count": nested_existing_count,
        "flat_existing_count": flat_existing_count,
        "missing_url_examples": missing_examples,
        "worker_clip_count_summary": _numeric_summary(worker_counts.values()),
    }


def _maybe_write_physical_source_manifest(
    config: dict[str, Any],
    *,
    source_root: Path,
    target_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    target_config = config.get("target", {})
    manifest_relpath = target_config.get("physical_source_manifest")
    if not manifest_relpath:
        return {"enabled": False}
    source_config = config.get("source", {})
    split = str(source_config.get("physical_source_split", "pretrain"))
    url_prefix = str(
        source_config.get(
            "physical_source_url_prefix",
            "https://storage.googleapis.com/buildai-imu-benchmark-v1-preexisting",
        )
    )
    max_urls = int(config["execution"].get("smoke_scan_files", 512)) if smoke else None
    urls = _physical_source_urls(
        source_root,
        split=split,
        url_prefix=url_prefix,
        max_urls=max_urls,
    )
    output_path = target_root / str(manifest_relpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{url}\n" for url in urls), encoding="utf-8")
    report = {
        "enabled": True,
        "mode": "smoke" if smoke else "full",
        "path": str(output_path),
        "split": split,
        "url_prefix": url_prefix,
        "url_count": len(urls),
        "worker_count": len({_worker_id_from_url(url) for url in urls}),
        "examples": urls[:10],
    }
    log_event("source_inventory", "physical_manifest_written", **report)
    return report


def _physical_source_urls(source_root: Path, *, split: str, url_prefix: str, max_urls: int | None) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    split_root = source_root / split
    if split_root.exists():
        for path in sorted(file_path for file_path in split_root.rglob("*.txt") if _is_data_txt(file_path)):
            relpath = path.relative_to(split_root).as_posix()
            url = f"{url_prefix.rstrip('/')}/{split}/{relpath}"
            if url not in seen:
                seen.add(url)
                urls.append(url)
            if max_urls is not None and len(urls) >= max_urls:
                return urls
    for path in sorted(file_path for file_path in source_root.glob("*.txt") if _is_data_txt(file_path)):
        parsed_url = _url_from_flat_source_name(path.name)
        if parsed_url and f"/{split}/" in parsed_url and parsed_url not in seen:
            seen.add(parsed_url)
            urls.append(parsed_url)
        if max_urls is not None and len(urls) >= max_urls:
            return urls
    return urls


def _is_data_txt(path: Path) -> bool:
    return path.suffix == ".txt" and not path.name.startswith("._")


def _url_from_flat_source_name(name: str) -> str:
    parts = name.split("__")
    if len(parts) >= 4 and parts[0] == "storage.googleapis.com":
        return "https://" + "/".join(parts)
    return ""


def _worker_from_source_path(path: Path, *, split: str) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part == split and index + 1 < len(parts) and parts[index + 1].startswith("worker"):
            return parts[index + 1]
    for part in parts:
        if part.startswith("worker"):
            return part
    return None


def _parse_flat_source_name(name: str) -> dict[str, str]:
    parts = name.split("__")
    result: dict[str, str] = {}
    for index, part in enumerate(parts):
        if part in {"pretrain", "val", "new"}:
            result["split"] = part
            if index + 1 < len(parts) and parts[index + 1].startswith("worker"):
                result["worker_id"] = parts[index + 1]
            break
    if "split" not in result:
        parsed = urlparse(name.replace("__", "/"))
        url_parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
        for part in url_parts:
            if part in {"pretrain", "val", "new"}:
                result["split"] = part
            if part.startswith("worker"):
                result["worker_id"] = part
    return result


def _numeric_summary(values: Any) -> dict[str, float]:
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


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Source inventory config must include a '{key}' object.")
    return value
