from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from marginal_value.data.split_manifest import hash_manifest_url, read_manifest_urls


class RegistryLeakageError(RuntimeError):
    """Raised when one physical clip appears in more than one active split."""


@dataclass(frozen=True)
class ClipRecord:
    sample_id: str
    split: str
    url: str
    source_group_id: str
    worker_id: str
    raw_path: Path
    feature_path: Path
    quality: Mapping[str, float] = field(default_factory=dict)

    @property
    def quality_score(self) -> float | None:
        if "quality_score" not in self.quality:
            return None
        return float(self.quality["quality_score"])


@dataclass(frozen=True)
class ClipRegistry:
    root: Path
    clips: tuple[ClipRecord, ...]
    by_sample_id: dict[str, ClipRecord] = field(init=False)

    def __post_init__(self) -> None:
        by_id = {clip.sample_id: clip for clip in self.clips}
        if len(by_id) != len(self.clips):
            duplicates = [sample_id for sample_id, count in Counter(clip.sample_id for clip in self.clips).items() if count > 1]
            preview = ", ".join(sorted(duplicates)[:5])
            raise RegistryLeakageError(f"Duplicate cached sample ids in registry: {preview}")
        object.__setattr__(self, "by_sample_id", by_id)

    def split_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(clip.split for clip in self.clips).items()))

    def clips_for_split(self, split: str) -> list[ClipRecord]:
        return [clip for clip in self.clips if clip.split == split]

    def source_group_counts(self, split: str | None = None) -> dict[str, int]:
        clips = self.clips if split is None else tuple(self.clips_for_split(split))
        return dict(sorted(Counter(clip.source_group_id for clip in clips).items()))

    def by_source_group(self, split: str | None = None) -> dict[str, list[ClipRecord]]:
        grouped: defaultdict[str, list[ClipRecord]] = defaultdict(list)
        clips = self.clips if split is None else tuple(self.clips_for_split(split))
        for clip in clips:
            grouped[clip.source_group_id].append(clip)
        return {group: sorted(rows, key=lambda row: row.sample_id) for group, rows in sorted(grouped.items())}


def build_clip_registry(
    root: str | Path,
    *,
    manifests: Mapping[str, str],
    feature_glob: str = "cache/features/*.npz",
    raw_glob: str = "cache/raw/*.jsonl",
    include_uncached: bool = False,
    quality_metadata_path: str | Path | None = None,
) -> ClipRegistry:
    root_path = Path(root)
    if not manifests:
        raise ValueError("At least one active registry manifest is required.")

    feature_by_id = _files_by_stem(root_path, feature_glob)
    raw_by_id = _files_by_stem(root_path, raw_glob)
    quality_by_id = _load_quality_metadata(root_path, quality_metadata_path)
    split_ids: dict[str, set[str]] = {}
    split_urls: dict[str, list[str]] = {}
    for split, manifest in manifests.items():
        split_name = str(split)
        urls = read_manifest_urls(root_path / str(manifest))
        ids = {hash_manifest_url(url) for url in urls}
        split_urls[split_name] = urls
        split_ids[split_name] = ids

    _validate_split_disjoint(split_ids)

    clips: list[ClipRecord] = []
    for split, urls in split_urls.items():
        for url in urls:
            sample_id = hash_manifest_url(url)
            raw_path = raw_by_id.get(sample_id, root_path / "cache" / "raw" / f"{sample_id}.jsonl")
            feature_path = feature_by_id.get(sample_id, root_path / "cache" / "features" / f"{sample_id}.npz")
            if not include_uncached and (sample_id not in raw_by_id or sample_id not in feature_by_id):
                continue
            clips.append(
                ClipRecord(
                    sample_id=sample_id,
                    split=split,
                    url=url,
                    source_group_id=source_group_id_from_url(url),
                    worker_id=worker_id_from_url(url),
                    raw_path=raw_path,
                    feature_path=feature_path,
                    quality=quality_by_id.get(sample_id, {}),
                )
            )
    return ClipRegistry(root=root_path, clips=tuple(clips))


def load_clip_registry_from_config(config: Mapping[str, Any]) -> ClipRegistry:
    data = _required_mapping(config, "data")
    return build_clip_registry(
        data["root"],
        manifests=_required_mapping(data, "manifests"),
        feature_glob=str(data.get("feature_glob", "cache/features/*.npz")),
        raw_glob=str(data.get("raw_glob", "cache/raw/*.jsonl")),
        include_uncached=bool(data.get("include_uncached", False)),
        quality_metadata_path=data.get("quality_metadata"),
    )


def audit_clip_registry_coverage(
    root: str | Path,
    *,
    manifests: Mapping[str, str],
    registry: ClipRegistry | None = None,
    feature_glob: str = "cache/features/*.npz",
    raw_glob: str = "cache/raw/*.jsonl",
) -> dict[str, dict[str, int | float | str]]:
    root_path = Path(root)
    feature_by_id = _files_by_stem(root_path, feature_glob)
    raw_by_id = _files_by_stem(root_path, raw_glob)
    registry_by_split: dict[str, list[ClipRecord]] = {}
    if registry is not None:
        for clip in registry.clips:
            registry_by_split.setdefault(clip.split, []).append(clip)

    by_alias: dict[str, dict[str, int | float | str]] = {}
    for split, manifest in sorted(manifests.items()):
        urls = read_manifest_urls(root_path / str(manifest))
        sample_ids = [hash_manifest_url(url) for url in urls]
        raw_present = [sample_id for sample_id in sample_ids if sample_id in raw_by_id]
        feature_present = [sample_id for sample_id in sample_ids if sample_id in feature_by_id]
        cached_both = [sample_id for sample_id in sample_ids if sample_id in raw_by_id and sample_id in feature_by_id]
        registry_clips = registry_by_split.get(str(split), [])
        registry_workers = {clip.worker_id for clip in registry_clips}
        registry_groups = {clip.source_group_id for clip in registry_clips}
        report = {
            "split": str(split),
            "manifest_url_count": int(len(urls)),
            "cached_url_count": int(len(cached_both)),
            "registry_clip_count": int(len(registry_clips)),
            "unique_workers": int(len(registry_workers)),
            "unique_source_groups": int(len(registry_groups)),
            "skipped_uncached_count": int(len(sample_ids) - len(cached_both)),
            "skipped_missing_raw_count": int(len(sample_ids) - len(raw_present)),
            "skipped_missing_feature_count": int(len(sample_ids) - len(feature_present)),
            "cached_fraction": _fraction(len(cached_both), len(sample_ids)),
        }
        by_alias[_coverage_alias(str(split))] = report
        by_alias[str(split)] = report
    return by_alias


def audit_clip_registry_coverage_from_config(
    config: Mapping[str, Any],
    *,
    registry: ClipRegistry | None = None,
) -> dict[str, dict[str, int | float | str]]:
    data = _required_mapping(config, "data")
    return audit_clip_registry_coverage(
        data["root"],
        manifests=_required_mapping(data, "manifests"),
        registry=registry,
        feature_glob=str(data.get("feature_glob", "cache/features/*.npz")),
        raw_glob=str(data.get("raw_glob", "cache/raw/*.jsonl")),
    )


def source_group_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
    for part in parts:
        if part.startswith("worker"):
            return part
    return parsed.netloc + ":" + PurePosixPath(parsed.path).name


def worker_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
    for part in parts:
        if part.startswith("worker"):
            return part
    return hash_manifest_url(url)


def _files_by_stem(root: Path, glob_pattern: str) -> dict[str, Path]:
    return {path.stem: path for path in root.glob(glob_pattern)}


def _validate_split_disjoint(split_ids: Mapping[str, set[str]]) -> None:
    split_names = sorted(split_ids)
    for left_idx, left in enumerate(split_names):
        for right in split_names[left_idx + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                raise RegistryLeakageError(
                    f"Active registry manifests '{left}' and '{right}' overlap on {len(overlap)} sample ids: {preview}"
                )


def _coverage_alias(split: str) -> str:
    if split == "pretrain":
        return "old"
    if split == "new":
        return "new"
    return split


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _load_quality_metadata(root: Path, quality_metadata_path: str | Path | None) -> dict[str, dict[str, float]]:
    if quality_metadata_path is None:
        return {}
    path = Path(quality_metadata_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Missing quality metadata: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return _quality_rows_to_lookup(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return _quality_rows_to_lookup(rows)


def _quality_rows_to_lookup(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        numeric: dict[str, float] = {}
        for key, value in row.items():
            if key == "sample_id":
                continue
            try:
                numeric[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        lookup[sample_id] = numeric
    return lookup


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"active registry config requires object field '{key}'.")
    return value
