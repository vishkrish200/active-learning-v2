from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


class LeakageError(RuntimeError):
    """Raised when a held-out split overlaps with the training split."""


@dataclass(frozen=True)
class SplitSample:
    sample_id: str
    split: str
    url: str
    raw_path: Path
    feature_path: Path

    def to_row(self, root: Path) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "url": self.url,
            "raw_path": _relative_or_abs(self.raw_path, root),
            "feature_path": _relative_or_abs(self.feature_path, root),
        }


def hash_manifest_url(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def read_manifest_urls(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
    return [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_split_manifest(
    root: str | Path,
    *,
    pretrain_manifest: str,
    val_manifest: str,
    extra_manifests: dict[str, str] | None = None,
    feature_glob: str,
    raw_glob: str,
) -> list[SplitSample]:
    root_path = Path(root)
    feature_by_id = {path.stem: path for path in root_path.glob(feature_glob)}
    raw_by_id = {path.stem: path for path in root_path.glob(raw_glob)}

    pretrain_urls = read_manifest_urls(root_path / pretrain_manifest)
    val_urls = read_manifest_urls(root_path / val_manifest)
    split_urls: dict[str, list[str]] = {"pretrain": pretrain_urls, "val": val_urls}
    for split, manifest in (extra_manifests or {}).items():
        if split in split_urls:
            raise ValueError(f"Duplicate split manifest configured for '{split}'.")
        split_urls[str(split)] = read_manifest_urls(root_path / manifest)

    split_ids = {split: {hash_manifest_url(url) for url in urls} for split, urls in split_urls.items()}
    split_names = sorted(split_ids)
    for left_idx, left in enumerate(split_names):
        for right in split_names[left_idx + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                raise LeakageError(f"{left} and {right} manifests overlap on {len(overlap)} sample ids: {preview}")

    rows: list[SplitSample] = []
    for split, urls in split_urls.items():
        rows.extend(_rows_for_split(split, urls, feature_by_id, raw_by_id))
    return rows


def select_split(manifest: Iterable[SplitSample], split: str) -> list[SplitSample]:
    selected = [row for row in manifest if row.split == split]
    if not selected:
        raise ValueError(f"No cached samples found for split '{split}'.")
    return selected


def split_counts(manifest: Iterable[SplitSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in manifest:
        counts[row.split] = counts.get(row.split, 0) + 1
    return counts


def write_manifest_csv(manifest: Iterable[SplitSample], root: str | Path, output_path: str | Path) -> None:
    root_path = Path(root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [sample.to_row(root_path) for sample in manifest]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "url", "raw_path", "feature_path"])
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_report(manifest: Iterable[SplitSample], output_path: str | Path) -> None:
    rows = list(manifest)
    report = {
        "split_counts": split_counts(rows),
        "n_samples": len(rows),
        "sample_ids_unique": len({row.sample_id for row in rows}) == len(rows),
        "raw_files_present": sum(row.raw_path.exists() for row in rows),
        "feature_files_present": sum(row.feature_path.exists() for row in rows),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _rows_for_split(
    split: str,
    urls: Iterable[str],
    feature_by_id: dict[str, Path],
    raw_by_id: dict[str, Path],
) -> list[SplitSample]:
    rows: list[SplitSample] = []
    for url in urls:
        sample_id = hash_manifest_url(url)
        feature_path = feature_by_id.get(sample_id)
        raw_path = raw_by_id.get(sample_id)
        if feature_path is None or raw_path is None:
            continue
        rows.append(
            SplitSample(
                sample_id=sample_id,
                split=split,
                url=url,
                raw_path=raw_path,
                feature_path=feature_path,
            )
        )
    return rows


def _relative_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
