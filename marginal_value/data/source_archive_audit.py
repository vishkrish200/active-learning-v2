from __future__ import annotations

import json
import tarfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import urlparse

from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls
from marginal_value.logging_utils import log_event, log_progress


def load_source_archive_audit_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_source_archive_audit_config(config: dict[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    source = _required_mapping(config, "source")
    target = _required_mapping(config, "target")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal":
        raise ValueError("Source archive audit must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(source.get("root", "")).startswith("/source"):
        raise ValueError("Source archive audit source.root must be mounted under /source.")
    if not allow_local_paths and not str(source.get("archive_path", "")).startswith("/source"):
        raise ValueError("Source archive audit source.archive_path must be mounted under /source.")
    if not allow_local_paths and not str(target.get("root", "")).startswith("/data"):
        raise ValueError("Source archive audit target.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("Source archive audit artifacts.output_dir must be mounted under /artifacts.")
    if not str(target.get("manifest", "")).startswith("cache/manifests/"):
        raise ValueError("target.manifest must be under cache/manifests/.")
    extracted_manifest = target.get("extracted_manifest")
    if extracted_manifest is not None and not str(extracted_manifest).startswith("cache/manifests/"):
        raise ValueError("target.extracted_manifest must be under cache/manifests/ when provided.")
    if int(execution.get("smoke_member_limit", 1)) <= 0:
        raise ValueError("execution.smoke_member_limit must be positive.")
    if int(execution.get("progress_every", 5000)) <= 0:
        raise ValueError("execution.progress_every must be positive.")
    if int(execution.get("full_member_progress_hint", 100000)) <= 0:
        raise ValueError("execution.full_member_progress_hint must be positive.")
    if not str(source.get("split", "")):
        raise ValueError("source.split must be non-empty.")
    if not str(source.get("url_prefix", "")):
        raise ValueError("source.url_prefix must be non-empty.")


def run_source_archive_audit(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_source_archive_audit_config(config)
    mode = "smoke" if smoke else "full"
    source_root = Path(config["source"]["root"])
    archive_path = Path(config["source"]["archive_path"])
    target_root = Path(config["target"]["root"])
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_urls = set(read_manifest_urls(target_root / str(config["target"]["manifest"])))
    extracted_manifest = config["target"].get("extracted_manifest")
    extracted_urls = (
        set(read_manifest_urls(target_root / str(extracted_manifest)))
        if extracted_manifest and (target_root / str(extracted_manifest)).exists()
        else set()
    )
    smoke_limit = int(config["execution"].get("smoke_member_limit", 2048))
    data_member_limit = smoke_limit if smoke else None
    progress_total = (
        data_member_limit
        if data_member_limit is not None
        else int(config["execution"].get("full_member_progress_hint", 100000))
    )
    progress_every = (
        max(1, data_member_limit // 10)
        if data_member_limit is not None
        else int(config["execution"].get("progress_every", 5000))
    )

    log_event(
        "source_archive_audit",
        "start",
        mode=mode,
        archive_path=str(archive_path),
        manifest_url_count=len(manifest_urls),
        extracted_url_count=len(extracted_urls),
    )
    archive_urls: list[str] = []
    member_count = 0
    data_member_count = 0
    metadata_member_count = 0
    skipped_member_count = 0
    conversion_failed_count = 0
    conversion_failed_examples: list[str] = []
    workers: Counter[str] = Counter()

    with _open_tar_archive(archive_path) as archive:
        for member in archive:
            member_count += 1
            if not member.isfile():
                skipped_member_count += 1
                continue
            url_result = _url_from_archive_member(
                member.name,
                split=str(config["source"]["split"]),
                url_prefix=str(config["source"]["url_prefix"]),
            )
            if url_result == "metadata":
                metadata_member_count += 1
                continue
            if not url_result:
                conversion_failed_count += 1
                if len(conversion_failed_examples) < 20:
                    conversion_failed_examples.append(member.name)
                continue
            archive_urls.append(url_result)
            data_member_count += 1
            workers[_worker_id_from_url(url_result)] += 1
            log_progress(
                "source_archive_audit",
                "member_progress",
                index=data_member_count,
                total=progress_total,
                every=progress_every,
                mode=mode,
                member_count=member_count,
            )
            if data_member_limit is not None and data_member_count >= data_member_limit:
                break

    archive_url_set = set(archive_urls)
    manifest_matches = archive_url_set & manifest_urls
    extracted_duplicates = archive_url_set & extracted_urls
    missing_from_extracted = manifest_matches - extracted_urls
    not_in_manifest = archive_url_set - manifest_urls
    member_urls_path = output_dir / str(config["artifacts"].get("member_urls_path", f"source_archive_members_{mode}.txt"))
    member_urls_path.write_text("".join(f"{url}\n" for url in archive_urls), encoding="utf-8")
    report = {
        "mode": mode,
        "source_root": str(source_root),
        "archive_path": str(archive_path),
        "manifest": {
            "path": str(target_root / str(config["target"]["manifest"])),
            "manifest_url_count": len(manifest_urls),
        },
        "extracted_manifest": {
            "path": str(target_root / str(extracted_manifest)) if extracted_manifest else "",
            "url_count": len(extracted_urls),
        },
        "archive": {
            "member_count": member_count,
            "data_member_count": data_member_count,
            "metadata_member_count": metadata_member_count,
            "skipped_member_count": skipped_member_count,
            "conversion_failed_count": conversion_failed_count,
            "conversion_failed_examples": conversion_failed_examples,
            "unique_url_count": len(archive_url_set),
            "duplicate_url_count": len(archive_urls) - len(archive_url_set),
            "worker_count": len(workers),
            "worker_clip_count_summary": _numeric_summary(workers.values()),
            "stopped_early": data_member_limit is not None and data_member_count >= data_member_limit,
            "member_urls_path": str(member_urls_path),
            "examples": archive_urls[:10],
        },
        "overlap": {
            "manifest_match_count": len(manifest_matches),
            "extracted_duplicate_count": len(extracted_duplicates),
            "missing_from_extracted_count": len(missing_from_extracted),
            "not_in_manifest_count": len(not_in_manifest),
            "missing_from_extracted_examples": sorted(missing_from_extracted)[:25],
            "not_in_manifest_examples": sorted(not_in_manifest)[:25],
        },
        "notes": [
            "This job lists archive members only; it does not extract archive payloads.",
            "missing_from_extracted_count is the key value: manifest URLs present in the archive but absent from the extracted physical-source manifest.",
        ],
    }
    report_path = output_dir / f"source_archive_audit_{mode}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "report_path": str(report_path),
        "archive_data_member_count": data_member_count,
        "archive_unique_url_count": len(archive_url_set),
        "archive_manifest_match_count": len(manifest_matches),
        "archive_extracted_duplicate_count": len(extracted_duplicates),
        "archive_missing_from_extracted_count": len(missing_from_extracted),
        "archive_not_in_manifest_count": len(not_in_manifest),
    }
    log_event("source_archive_audit", "done", **result)
    return result


@contextmanager
def _open_tar_archive(path: Path) -> Iterator[tarfile.TarFile]:
    if path.suffix == ".zst" or path.name.endswith(".tar.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Reading .tar.zst archives requires the zstandard package.") from exc
        with path.open("rb") as compressed:
            reader = zstd.ZstdDecompressor().stream_reader(compressed)
            try:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    yield archive
            finally:
                reader.close()
        return
    with tarfile.open(path, "r:*") as archive:
        yield archive


def _url_from_archive_member(name: str, *, split: str, url_prefix: str) -> str:
    path = PurePosixPath(str(name))
    parts = [part for part in path.parts if part not in {"", ".", "/"}]
    if not parts:
        return ""
    basename = parts[-1]
    if basename.startswith("._"):
        return "metadata"
    if not basename.endswith(".txt"):
        return ""
    flat_url = _url_from_flat_source_name(basename)
    if flat_url:
        return flat_url
    if split in parts:
        split_index = parts.index(split)
        remainder = parts[split_index + 1 :]
        if remainder:
            return f"{url_prefix.rstrip('/')}/{split}/{'/'.join(remainder)}"
    for index, part in enumerate(parts):
        if part.startswith("worker"):
            return f"{url_prefix.rstrip('/')}/{split}/{'/'.join(parts[index:])}"
    return f"{url_prefix.rstrip('/')}/{split}/{basename}"


def _url_from_flat_source_name(name: str) -> str:
    parts = name.split("__")
    if len(parts) >= 4 and parts[0] == "storage.googleapis.com":
        return "https://" + "/".join(parts)
    parsed = urlparse(name.replace("__", "/"))
    if parsed.netloc == "storage.googleapis.com":
        return "https://" + parsed.netloc + parsed.path
    return ""


def _worker_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
    for part in parts:
        if part.startswith("worker"):
            return part
    return hash_manifest_url(url)


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


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Source archive audit config must include a '{key}' object.")
    return value
