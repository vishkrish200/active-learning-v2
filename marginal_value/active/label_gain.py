from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.active.evaluate_active_loop import (
    DEFAULT_REPRESENTATIONS,
    _oracle_greedy_order,
    coverage_gain_for_ids,
    load_active_episodes,
)
from marginal_value.active.embedding_cache import (
    SUPPORTED_REPRESENTATIONS,
    embedding_cache_dir_from_config,
    load_embedding_lookup,
)
from marginal_value.active.registry import (
    ClipRegistry,
    audit_clip_registry_coverage_from_config,
    load_clip_registry_from_config,
)
from marginal_value.indexing.knn_features import normalize_rows
from marginal_value.logging_utils import log_event, log_progress


def run_active_label_gain(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_active_label_gain_config(config)
    mode = "smoke" if smoke else "full"
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / f"active_label_gain_{mode}.jsonl"
    labels_csv_path = output_dir / f"active_label_gain_{mode}.csv"
    report_path = output_dir / f"active_label_gain_report_{mode}.json"
    log_event("active_label_gain", "start", mode=mode)

    registry = load_clip_registry_from_config(config)
    registry_coverage = audit_clip_registry_coverage_from_config(config, registry=registry)
    episodes = load_active_episodes(_episode_path(config))
    if smoke:
        episodes = episodes[: int(config["execution"].get("smoke_max_episodes", 1))]
    if not episodes:
        raise ValueError("Active label-gain generation requires at least one episode.")

    label_config = config["labels"]
    representations = [str(rep) for rep in label_config.get("representations", DEFAULT_REPRESENTATIONS)]
    used_ids = _episode_sample_ids(episodes)
    missing = sorted(sample_id for sample_id in used_ids if sample_id not in registry.by_sample_id)
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Episodes reference {len(missing)} clips absent from the active registry: {preview}")

    embedding_result = load_embedding_lookup(
        [registry.by_sample_id[sample_id] for sample_id in sorted(used_ids)],
        representations=representations,
        sample_rate=float(label_config.get("sample_rate", 30.0)),
        raw_shape_max_samples=label_config.get("raw_shape_max_samples"),
        cache_dir=embedding_cache_dir_from_config(config),
        component="active_label_gain",
        mode=mode,
        representation_options=_representation_options(config, label_config),
    )
    embeddings = embedding_result.embeddings

    rows: list[dict[str, object]] = []
    progress_every = max(1, len(episodes) // 10)
    for episode_index, episode in enumerate(episodes, start=1):
        rows.extend(_label_episode_candidates(episode=episode, registry=registry, embeddings=embeddings, representations=representations))
        log_progress(
            "active_label_gain",
            "episode_progress",
            index=episode_index,
            total=len(episodes),
            every=progress_every,
            mode=mode,
            labels=len(rows),
        )
    rows = compute_gated_labels(rows, **dict(label_config.get("hygiene_gate", {})))

    _write_jsonl(labels_path, rows)
    _write_csv(labels_csv_path, rows)
    report = {
        "mode": mode,
        "n_episodes": len(episodes),
        "n_labels": len(rows),
        "representations": representations,
        "registry_coverage": _deduplicate_coverage_aliases(registry_coverage),
        "registry_coverage_summary": _registry_coverage_summary(registry_coverage),
        "embedding_cache": embedding_result.report(),
        "label_summary": _label_summary(rows),
        "artifacts": {
            "labels_jsonl": str(labels_path),
            "labels_csv": str(labels_csv_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "n_episodes": len(episodes),
        "n_labels": len(rows),
        "labels_path": str(labels_path),
        "labels_csv_path": str(labels_csv_path),
        "report_path": str(report_path),
    }
    log_event("active_label_gain", "done", **result)
    return result


def validate_active_label_gain_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    episodes = _required_mapping(config, "episodes")
    labels = _required_mapping(config, "labels")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Active label gain must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    if "path" not in episodes:
        raise ValueError("episodes.path is required.")
    if not manifests:
        raise ValueError("data.manifests must not be empty.")
    for manifest in manifests.values():
        if not str(manifest).startswith("cache/manifests/"):
            raise ValueError("data.manifests paths must be under cache/manifests/.")
    representations = labels.get("representations", DEFAULT_REPRESENTATIONS)
    if not isinstance(representations, list | tuple) or not representations:
        raise ValueError("labels.representations must be a non-empty list.")
    unsupported = set(str(rep) for rep in representations) - set(SUPPORTED_REPRESENTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported active-label representations: {sorted(unsupported)}")
    if int(execution.get("smoke_max_episodes", 1)) <= 0:
        raise ValueError("execution.smoke_max_episodes must be positive.")


def _label_episode_candidates(
    *,
    episode: Any,
    registry: ClipRegistry,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    representations: Sequence[str],
) -> list[dict[str, object]]:
    candidate_ids = list(episode.candidate_clip_ids)
    primary_representation = str(representations[0])
    new_cluster_similarity = _candidate_nearest_similarity(embeddings[primary_representation], candidate_ids)
    greedy_order = _oracle_greedy_order(
        episode=episode,
        embeddings=embeddings,
        representations=representations,
        max_selected=len(candidate_ids),
    )
    greedy_rank_by_idx = {int(candidate_idx): rank for rank, candidate_idx in enumerate(greedy_order, start=1)}
    greedy_prefix_by_idx: dict[int, list[str]] = {}
    prefix: list[str] = []
    for candidate_idx in greedy_order:
        greedy_prefix_by_idx[int(candidate_idx)] = list(prefix)
        prefix.append(candidate_ids[int(candidate_idx)])

    rows: list[dict[str, object]] = []
    for candidate_idx, sample_id in enumerate(candidate_ids):
        clip = registry.by_sample_id[sample_id]
        alone_metrics = {
            rep: coverage_gain_for_ids(
                support_clip_ids=episode.support_clip_ids,
                hidden_target_clip_ids=episode.hidden_target_clip_ids,
                selected_clip_ids=[sample_id],
                embeddings=embeddings[rep],
            )
            for rep in representations
        }
        prefix_ids = greedy_prefix_by_idx.get(candidate_idx, [])
        prefix_metrics = {
            rep: coverage_gain_for_ids(
                support_clip_ids=[*episode.support_clip_ids, *prefix_ids],
                hidden_target_clip_ids=episode.hidden_target_clip_ids,
                selected_clip_ids=[sample_id],
                embeddings=embeddings[rep],
            )
            for rep in representations
        }
        alone_balanced = _balanced_metrics(alone_metrics.values())
        prefix_balanced = _balanced_metrics(prefix_metrics.values())
        row: dict[str, object] = {
            "episode_id": episode.episode_id,
            "candidate_index": int(candidate_idx),
            "sample_id": sample_id,
            "worker_id": clip.worker_id,
            "source_group_id": clip.source_group_id,
            "candidate_role": str(episode.candidate_roles.get(sample_id, "unlabeled")),
            "quality_score": _quality_value(clip, "quality_score", default=1.0),
            "stationary_fraction": _quality_value(clip, "stationary_fraction", default=0.0),
            "max_abs_value": _quality_value(clip, "max_abs_value", default=0.0),
            "new_cluster_similarity": float(new_cluster_similarity[candidate_idx]),
            "gain_if_added_alone": float(alone_balanced["absolute_gain"]),
            "relative_gain_if_added_alone": float(alone_balanced["relative_gain"]),
            "balanced_gain": float(alone_balanced["absolute_gain"]),
            "balanced_relative_gain": float(alone_balanced["relative_gain"]),
            "greedy_prefix_rank": int(greedy_rank_by_idx.get(candidate_idx, 0)),
            "greedy_prefix_size": int(len(prefix_ids)),
            "gain_after_greedy_prefix": float(prefix_balanced["absolute_gain"]),
            "relative_gain_after_greedy_prefix": float(prefix_balanced["relative_gain"]),
            "balanced_gain_after_greedy_prefix": float(prefix_balanced["absolute_gain"]),
            "balanced_relative_gain_after_greedy_prefix": float(prefix_balanced["relative_gain"]),
        }
        for rep, metrics in alone_metrics.items():
            suffix = rep
            row[f"coverage_before_{suffix}"] = float(metrics["coverage_before"])
            row[f"coverage_after_{suffix}"] = float(metrics["coverage_after"])
            row[f"gain_{suffix}"] = float(metrics["absolute_gain"])
            row[f"relative_gain_{suffix}"] = float(metrics["relative_gain"])
        for rep, metrics in prefix_metrics.items():
            suffix = rep
            row[f"gain_after_greedy_prefix_{suffix}"] = float(metrics["absolute_gain"])
            row[f"relative_gain_after_greedy_prefix_{suffix}"] = float(metrics["relative_gain"])
        rows.append(row)
    return rows


def compute_gated_labels(
    label_rows: Sequence[Mapping[str, object]],
    *,
    quality_threshold: float = 0.85,
    max_stationary_fraction: float = 0.90,
    max_abs_value: float = 60.0,
    duplicate_cosine_threshold: float | None = 0.985,
    component: str = "active_label_gain",
) -> list[dict[str, object]]:
    if not label_rows:
        return []
    required = {
        "balanced_gain",
        "balanced_relative_gain",
        "quality_score",
        "stationary_fraction",
        "max_abs_value",
    }
    available = {key for row in label_rows for key in row}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Cannot compute gated labels; missing required columns: {missing}")
    duplicate_column = None
    if duplicate_cosine_threshold is not None:
        duplicate_column = _first_available_column(
            available,
            (
                "new_cluster_similarity",
                "candidate_nn_similarity",
                "nearest_candidate_similarity",
            ),
        )
    if duplicate_column is None and duplicate_cosine_threshold is not None:
        log_event(
            component,
            "hygiene_gate_duplicate_similarity_missing",
            warning="duplicate similarity column unavailable; duplicate gate skipped",
        )

    gated: list[dict[str, object]] = []
    failure_counts = {
        "quality": 0,
        "stationary": 0,
        "abs_value": 0,
        "duplicate": 0,
    }
    for row in label_rows:
        quality_ok = _safe_float(row.get("quality_score"), 0.0) >= float(quality_threshold)
        stationary_ok = _safe_float(row.get("stationary_fraction"), 1.0) <= float(max_stationary_fraction)
        abs_value_ok = _safe_float(row.get("max_abs_value"), float("inf")) <= float(max_abs_value)
        duplicate_ok = True
        if duplicate_column is not None:
            duplicate_ok = _safe_float(row.get(duplicate_column), 0.0) <= float(duplicate_cosine_threshold)
        failure_counts["quality"] += int(not quality_ok)
        failure_counts["stationary"] += int(not stationary_ok)
        failure_counts["abs_value"] += int(not abs_value_ok)
        failure_counts["duplicate"] += int(not duplicate_ok)
        hygiene_gate = bool(quality_ok and stationary_ok and abs_value_ok and duplicate_ok)
        multiplier = 1.0 if hygiene_gate else 0.0
        output = dict(row)
        output["hygiene_gate"] = hygiene_gate
        output["gated_balanced_gain"] = float(_safe_float(row.get("balanced_gain"), 0.0) * multiplier)
        output["gated_balanced_relative_gain"] = float(_safe_float(row.get("balanced_relative_gain"), 0.0) * multiplier)
        if "balanced_gain_after_greedy_prefix" in row:
            output["gated_balanced_gain_after_greedy_prefix"] = float(
                _safe_float(row.get("balanced_gain_after_greedy_prefix"), 0.0) * multiplier
            )
        if "balanced_relative_gain_after_greedy_prefix" in row:
            output["gated_balanced_relative_gain_after_greedy_prefix"] = float(
                _safe_float(row.get("balanced_relative_gain_after_greedy_prefix"), 0.0) * multiplier
            )
        gated.append(output)

    n_total = len(gated)
    n_failed = int(sum(not bool(row.get("hygiene_gate", False)) for row in gated))
    log_event(
        component,
        "hygiene_gate",
        n_total=n_total,
        n_failed=n_failed,
        failed_fraction=float(n_failed / n_total) if n_total else 0.0,
        quality_failures=int(failure_counts["quality"]),
        stationary_failures=int(failure_counts["stationary"]),
        abs_value_failures=int(failure_counts["abs_value"]),
        duplicate_failures=int(failure_counts["duplicate"]),
        duplicate_similarity_column=duplicate_column or "",
        quality_threshold=float(quality_threshold),
        max_stationary_fraction=float(max_stationary_fraction),
        max_abs_value=float(max_abs_value),
        duplicate_cosine_threshold=float(duplicate_cosine_threshold) if duplicate_cosine_threshold is not None else "",
    )
    return gated


def _balanced_metrics(metrics: Sequence[Mapping[str, float]] | Any) -> dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {"absolute_gain": 0.0, "relative_gain": 0.0}
    return {
        "absolute_gain": float(np.mean([float(row["absolute_gain"]) for row in rows])),
        "relative_gain": float(np.mean([float(row["relative_gain"]) for row in rows])),
    }


def _label_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "candidate_count": int(len(rows)),
        "positive_balanced_gain_count": int(sum(float(row.get("balanced_gain", 0.0)) > 1.0e-12 for row in rows)),
        "positive_gated_balanced_gain_count": int(sum(float(row.get("gated_balanced_gain", 0.0)) > 1.0e-12 for row in rows)),
        "hygiene_gate_pass_count": int(sum(bool(row.get("hygiene_gate", False)) for row in rows)),
        "hygiene_gate_fail_count": int(sum(not bool(row.get("hygiene_gate", False)) for row in rows)),
        "candidate_role_counts": dict(sorted(_counts(str(row.get("candidate_role", "unlabeled")) for row in rows).items())),
        "balanced_gain": _numeric_summary(float(row.get("balanced_gain", 0.0)) for row in rows),
        "balanced_relative_gain": _numeric_summary(float(row.get("balanced_relative_gain", 0.0)) for row in rows),
        "gated_balanced_gain": _numeric_summary(float(row.get("gated_balanced_gain", 0.0)) for row in rows),
        "gated_balanced_relative_gain": _numeric_summary(float(row.get("gated_balanced_relative_gain", 0.0)) for row in rows),
        "gain_after_greedy_prefix": _numeric_summary(float(row.get("gain_after_greedy_prefix", 0.0)) for row in rows),
    }


def _candidate_nearest_similarity(embeddings: Mapping[str, np.ndarray], candidate_ids: Sequence[str]) -> np.ndarray:
    if not candidate_ids:
        return np.asarray([], dtype=float)
    vectors = normalize_rows(np.vstack([embeddings[str(sample_id)] for sample_id in candidate_ids]))
    if len(vectors) <= 1:
        return np.zeros(len(vectors), dtype=float)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = np.max(similarities, axis=1)
    return np.where(np.isfinite(nearest), nearest, 0.0)


def _episode_path(config: Mapping[str, Any]) -> Path:
    path = Path(str(config["episodes"]["path"]))
    if path.is_absolute():
        return path
    return Path(str(config["data"]["root"])) / path


def _episode_sample_ids(episodes: Sequence[Any]) -> set[str]:
    sample_ids: set[str] = set()
    for episode in episodes:
        sample_ids.update(episode.support_clip_ids)
        sample_ids.update(episode.candidate_clip_ids)
        sample_ids.update(episode.hidden_target_clip_ids)
    return sample_ids


def _quality_value(clip: Any, key: str, *, default: float) -> float:
    try:
        return float(clip.quality.get(key, default))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _first_available_column(available: set[str], names: Sequence[str]) -> str | None:
    for name in names:
        if name in available:
            return name
    return None


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_summary(values: Any) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(numbers)),
        "mean": float(np.mean(numbers)),
        "min": float(min(numbers)),
        "max": float(max(numbers)),
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _deduplicate_coverage_aliases(coverage: Mapping[str, Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    seen_ids: set[int] = set()
    for key in ("old", "new"):
        if key in coverage:
            output[key] = coverage[key]
            seen_ids.add(id(coverage[key]))
    for key, value in coverage.items():
        if key not in output and id(value) not in seen_ids:
            output[key] = value
    return output


def _registry_coverage_summary(coverage: Mapping[str, Mapping[str, object]]) -> dict[str, int]:
    old = coverage.get("old", {})
    new = coverage.get("new", {})
    return {
        "manifest_old_url_count": int(old.get("manifest_url_count", 0)),
        "cached_old_url_count": int(old.get("cached_url_count", 0)),
        "registry_old_clip_count": int(old.get("registry_clip_count", 0)),
        "unique_old_workers": int(old.get("unique_workers", 0)),
        "unique_old_source_groups": int(old.get("unique_source_groups", 0)),
        "manifest_new_url_count": int(new.get("manifest_url_count", 0)),
        "cached_new_url_count": int(new.get("cached_url_count", 0)),
        "registry_new_clip_count": int(new.get("registry_clip_count", 0)),
        "unique_new_workers": int(new.get("unique_workers", 0)),
        "skipped_uncached_count": int(old.get("skipped_uncached_count", 0)) + int(new.get("skipped_uncached_count", 0)),
        "skipped_missing_raw_count": int(old.get("skipped_missing_raw_count", 0)) + int(new.get("skipped_missing_raw_count", 0)),
        "skipped_missing_feature_count": int(old.get("skipped_missing_feature_count", 0)) + int(new.get("skipped_missing_feature_count", 0)),
    }


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"active label-gain config requires object field '{key}'.")
    return value


def _representation_options(config: Mapping[str, Any], section: Mapping[str, Any]) -> Mapping[str, object]:
    options: dict[str, object] = {}
    embeddings = config.get("embeddings")
    if isinstance(embeddings, Mapping):
        options.update(dict(embeddings))
    explicit = section.get("representation_options")
    if isinstance(explicit, Mapping):
        options.update(dict(explicit))
    return options
