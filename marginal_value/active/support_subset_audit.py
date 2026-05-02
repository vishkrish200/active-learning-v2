from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.active.registry import ClipRecord, load_clip_registry_from_config
from marginal_value.logging_utils import log_event


DEFAULT_AUDIT_FIELDS = ("quality_score", "stationary_fraction", "max_abs_value")


def run_support_subset_audit(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Audit representativeness of the budgeted old-support subsets."""
    validate_support_subset_audit_config(config)
    mode = "smoke" if smoke else "full"
    ranking = config["ranking"]
    output_dir = Path(str(config["artifacts"]["output_dir"]))
    audit_dir = output_dir / "support_subset_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_event("support_subset_audit", "start", mode=mode)

    registry = load_clip_registry_from_config(config)
    support_split = str(ranking.get("support_split", "pretrain"))
    full_support = sorted(registry.clips_for_split(support_split), key=lambda clip: clip.sample_id)
    if not full_support:
        raise ValueError(f"No support clips available for split '{support_split}'.")
    by_id = {clip.sample_id: clip for clip in full_support}

    left_ids = _load_partial_shard_sample_ids(
        Path(str(ranking["left_support_shard_dir"])),
        allowed_sample_ids=set(by_id),
        max_shards=_optional_int(ranking.get("left_support_max_shards")),
    )
    left_support = [by_id[sample_id] for sample_id in left_ids if sample_id in by_id]

    right_support = _right_support_subset(full_support, ranking=ranking)
    random_support = _deterministic_random_subset(
        full_support,
        size=len(right_support),
        seed=int(config.get("support_audit", {}).get("random_seed", 20260502))
        if isinstance(config.get("support_audit"), Mapping)
        else 20260502,
    )
    if smoke:
        limit = int(config["execution"].get("smoke_support_samples", 512))
        full_support_for_report = full_support[:limit]
        left_support = left_support[:limit]
        right_support = right_support[:limit]
        random_support = random_support[:limit]
    else:
        full_support_for_report = full_support

    summaries = {
        "full_support": _subset_summary(full_support_for_report, full_support=full_support),
        "partial_ts2vec_support": _subset_summary(left_support, full_support=full_support),
        "window_support_subset": _subset_summary(right_support, full_support=full_support),
        "random_window_size_support": _subset_summary(random_support, full_support=full_support),
    }
    source_rows = _source_group_rows(
        full_support=full_support,
        partial_ts2vec_support=left_support,
        window_support_subset=right_support,
        random_window_size_support=random_support,
    )
    worker_rows = _worker_rows(
        full_support=full_support,
        partial_ts2vec_support=left_support,
        window_support_subset=right_support,
        random_window_size_support=random_support,
    )

    source_path = audit_dir / f"support_subset_audit_source_groups_{mode}.csv"
    worker_path = audit_dir / f"support_subset_audit_workers_{mode}.csv"
    report_path = audit_dir / f"support_subset_audit_report_{mode}.json"
    _write_rows(source_path, source_rows)
    _write_rows(worker_path, worker_rows)
    report = {
        "mode": mode,
        "support_split": support_split,
        "left_support_shard_dir": str(ranking["left_support_shard_dir"]),
        "right_support_max_clips": None
        if ranking.get("right_support_max_clips") is None
        else int(ranking["right_support_max_clips"]),
        "summaries": summaries,
        "artifacts": {
            "report": str(report_path),
            "source_groups": str(source_path),
            "workers": str(worker_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "report_path": str(report_path),
        "source_groups_path": str(source_path),
        "workers_path": str(worker_path),
        "n_full_support": len(full_support),
        "n_partial_ts2vec_support": len(left_support),
        "n_window_support": len(right_support),
    }
    log_event("support_subset_audit", "done", **result)
    return result


def validate_support_subset_audit_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    ranking = _required_mapping(config, "ranking")
    if execution.get("provider") != "modal":
        raise ValueError("Support subset audit must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    if not str(ranking.get("left_support_shard_dir", "")).strip():
        raise ValueError("ranking.left_support_shard_dir is required.")
    if ranking.get("right_support_max_clips") is not None and int(ranking["right_support_max_clips"]) <= 0:
        raise ValueError("ranking.right_support_max_clips must be positive when provided.")


def _load_partial_shard_sample_ids(
    shard_dir: Path,
    *,
    allowed_sample_ids: set[str],
    max_shards: int | None,
) -> list[str]:
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if max_shards is not None:
        shard_paths = shard_paths[: int(max_shards)]
    if not shard_paths:
        raise FileNotFoundError(f"No support shard files found in {shard_dir}.")
    sample_ids: list[str] = []
    seen: set[str] = set()
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            for sample_id in [str(value) for value in data["sample_ids"].tolist()]:
                if sample_id in allowed_sample_ids and sample_id not in seen:
                    sample_ids.append(sample_id)
                    seen.add(sample_id)
    if not sample_ids:
        raise ValueError(f"Support shards in {shard_dir} contain no sample IDs from the support split.")
    return sample_ids


def _right_support_subset(clips: Sequence[ClipRecord], *, ranking: Mapping[str, Any]) -> list[ClipRecord]:
    max_support = ranking.get("right_support_max_clips")
    if max_support is None:
        return list(clips)
    return list(clips[: int(max_support)])


def _deterministic_random_subset(clips: Sequence[ClipRecord], *, size: int, seed: int) -> list[ClipRecord]:
    if size >= len(clips):
        return list(clips)
    rng = np.random.default_rng(int(seed))
    indices = sorted(int(index) for index in rng.choice(len(clips), size=int(size), replace=False))
    return [clips[index] for index in indices]


def _subset_summary(clips: Sequence[ClipRecord], *, full_support: Sequence[ClipRecord]) -> dict[str, object]:
    workers = Counter(clip.worker_id for clip in clips)
    groups = Counter(clip.source_group_id for clip in clips)
    full_workers = {clip.worker_id for clip in full_support}
    full_groups = {clip.source_group_id for clip in full_support}
    quality_fields = {
        field: _numeric_summary([float(clip.quality[field]) for clip in clips if field in clip.quality])
        for field in DEFAULT_AUDIT_FIELDS
    }
    quality_present = sum(bool(clip.quality) for clip in clips)
    return {
        "n_clips": int(len(clips)),
        "unique_workers": int(len(workers)),
        "unique_source_groups": int(len(groups)),
        "worker_coverage_vs_full": _fraction(len(workers), len(full_workers)),
        "source_group_coverage_vs_full": _fraction(len(groups), len(full_groups)),
        "top_worker_fraction": _top_fraction(workers),
        "top_source_group_fraction": _top_fraction(groups),
        "top5_worker_fraction": _topn_fraction(workers, 5),
        "top5_source_group_fraction": _topn_fraction(groups, 5),
        "quality_metadata_present_count": int(quality_present),
        "quality_metadata_present_fraction": _fraction(quality_present, len(clips)),
        "quality": quality_fields,
    }


def _source_group_rows(
    *,
    full_support: Sequence[ClipRecord],
    partial_ts2vec_support: Sequence[ClipRecord],
    window_support_subset: Sequence[ClipRecord],
    random_window_size_support: Sequence[ClipRecord],
) -> list[dict[str, object]]:
    return _count_rows(
        "source_group_id",
        {
            "full_support": Counter(clip.source_group_id for clip in full_support),
            "partial_ts2vec_support": Counter(clip.source_group_id for clip in partial_ts2vec_support),
            "window_support_subset": Counter(clip.source_group_id for clip in window_support_subset),
            "random_window_size_support": Counter(clip.source_group_id for clip in random_window_size_support),
        },
    )


def _worker_rows(
    *,
    full_support: Sequence[ClipRecord],
    partial_ts2vec_support: Sequence[ClipRecord],
    window_support_subset: Sequence[ClipRecord],
    random_window_size_support: Sequence[ClipRecord],
) -> list[dict[str, object]]:
    return _count_rows(
        "worker_id",
        {
            "full_support": Counter(clip.worker_id for clip in full_support),
            "partial_ts2vec_support": Counter(clip.worker_id for clip in partial_ts2vec_support),
            "window_support_subset": Counter(clip.worker_id for clip in window_support_subset),
            "random_window_size_support": Counter(clip.worker_id for clip in random_window_size_support),
        },
    )


def _count_rows(key: str, counters: Mapping[str, Counter[str]]) -> list[dict[str, object]]:
    full = counters["full_support"]
    labels = list(counters)
    rows = []
    for value in sorted(full):
        row: dict[str, object] = {key: value}
        full_count = int(full[value])
        for label in labels:
            count = int(counters[label].get(value, 0))
            row[f"{label}_count"] = count
            row[f"{label}_fraction_of_full"] = _fraction(count, full_count)
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    if rows:
        first_key = next(iter(rows[0].keys()))
        fieldnames = [first_key, *[field for field in fieldnames if field != first_key]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p25": None, "p50": None, "p75": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def _top_fraction(counter: Counter[str]) -> float:
    if not counter:
        return 0.0
    return float(max(counter.values()) / sum(counter.values()))


def _topn_fraction(counter: Counter[str], n: int) -> float:
    if not counter:
        return 0.0
    return float(sum(count for _key, count in counter.most_common(int(n))) / sum(counter.values()))


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"support subset audit config requires object field '{key}'.")
    return value
