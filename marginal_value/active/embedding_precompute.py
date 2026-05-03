from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from marginal_value.active.embedding_cache import SUPPORTED_REPRESENTATIONS, write_embedding_shard_cache
from marginal_value.active.evaluate_active_loop import load_active_episodes
from marginal_value.active.registry import (
    audit_clip_registry_coverage_from_config,
    load_clip_registry_from_config,
)
from marginal_value.logging_utils import log_event


def run_active_embedding_precompute(
    config: dict[str, Any],
    *,
    smoke: bool = False,
    on_shard_written: Callable[[], None] | None = None,
) -> dict[str, Any]:
    validate_active_embedding_precompute_config(config)
    mode = "smoke" if smoke else "full"
    log_event("active_embedding_precompute", "start", mode=mode)

    registry = load_clip_registry_from_config(config)
    registry_coverage = audit_clip_registry_coverage_from_config(config, registry=registry)
    episodes = load_active_episodes(_episode_path(config))
    if smoke:
        episodes = episodes[: int(config["execution"].get("smoke_max_episodes", 1))]
    if not episodes:
        raise ValueError("Active embedding precompute requires at least one episode.")

    sample_ids = sorted(_episode_sample_ids(episodes))
    missing = [sample_id for sample_id in sample_ids if sample_id not in registry.by_sample_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Episodes reference {len(missing)} clips absent from the active registry: {preview}")
    clips = [registry.by_sample_id[sample_id] for sample_id in sample_ids]

    precompute = config["precompute"]
    result = write_embedding_shard_cache(
        clips,
        representations=[str(rep) for rep in precompute.get("representations", SUPPORTED_REPRESENTATIONS)],
        sample_rate=float(precompute.get("sample_rate", 30.0)),
        raw_shape_max_samples=precompute.get("raw_shape_max_samples"),
        cache_dir=Path(str(config["embeddings"]["cache_dir"])),
        shard_size=int(precompute.get("shard_size", 512)),
        component="active_embedding_precompute",
        mode=mode,
        on_shard_written=on_shard_written,
        workers=int(precompute.get("workers", 1)),
    )

    report_path = Path(str(config["embeddings"]["cache_dir"])) / f"active_embedding_precompute_report_{mode}.json"
    report = {
        "mode": mode,
        "n_episodes": len(episodes),
        "n_clips": len(clips),
        "representations": [str(rep) for rep in precompute.get("representations", SUPPORTED_REPRESENTATIONS)],
        "shard_size": int(precompute.get("shard_size", 512)),
        "workers": int(precompute.get("workers", 1)),
        "registry_coverage": _deduplicate_coverage_aliases(registry_coverage),
        "registry_coverage_summary": _registry_coverage_summary(registry_coverage),
        "embedding_cache": result,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output = {
        "mode": mode,
        "n_episodes": len(episodes),
        "n_clips": len(clips),
        "n_shards": int(result["n_shards"]),
        "manifest_path": str(result["manifest_path"]),
        "report_path": str(report_path),
    }
    log_event("active_embedding_precompute", "done", **output)
    return output


def validate_active_embedding_precompute_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    embeddings = _required_mapping(config, "embeddings")
    episodes = _required_mapping(config, "episodes")
    precompute = _required_mapping(config, "precompute")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Active embedding precompute must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(embeddings.get("cache_dir", "")).startswith("/artifacts"):
        raise ValueError("embeddings.cache_dir must be mounted under /artifacts.")
    if "path" not in episodes:
        raise ValueError("episodes.path is required.")
    if not manifests:
        raise ValueError("data.manifests must not be empty.")
    for manifest in manifests.values():
        if not str(manifest).startswith("cache/manifests/"):
            raise ValueError("data.manifests paths must be under cache/manifests/.")
    representations = precompute.get("representations", SUPPORTED_REPRESENTATIONS)
    if not isinstance(representations, list | tuple) or not representations:
        raise ValueError("precompute.representations must be a non-empty list.")
    unsupported = set(str(rep) for rep in representations) - set(SUPPORTED_REPRESENTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported active embedding representations: {sorted(unsupported)}")
    if int(precompute.get("shard_size", 512)) <= 0:
        raise ValueError("precompute.shard_size must be positive.")
    if int(precompute.get("workers", 1)) <= 0:
        raise ValueError("precompute.workers must be positive.")
    if int(execution.get("smoke_max_episodes", 1)) <= 0:
        raise ValueError("execution.smoke_max_episodes must be positive.")


def _episode_path(config: Mapping[str, Any]) -> Path:
    path = Path(str(config["episodes"]["path"]))
    if path.is_absolute():
        return path
    return Path(str(config["data"]["root"])) / path


def _episode_sample_ids(episodes: object) -> set[str]:
    sample_ids: set[str] = set()
    for episode in episodes:
        sample_ids.update(episode.support_clip_ids)
        sample_ids.update(episode.candidate_clip_ids)
        sample_ids.update(episode.hidden_target_clip_ids)
    return sample_ids


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
        raise ValueError(f"active embedding precompute config requires object field '{key}'.")
    return value
