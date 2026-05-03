from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.active.registry import ClipRecord, ClipRegistry, load_clip_registry_from_config
from marginal_value.logging_utils import log_event


class EpisodeGenerationError(RuntimeError):
    """Raised when a source-blocked active-acquisition episode cannot be built."""


@dataclass(frozen=True)
class EpisodeConfig:
    n_episodes: int = 1
    seed: int = 17
    split: str = "pretrain"
    support_source_groups_per_episode: int = 128
    heldout_source_groups_per_episode: int = 32
    support_clips_per_group: int = 12
    known_like_candidates_per_group: int = 1
    heldout_candidates_per_group: int = 1
    hidden_targets_per_group: int = 2
    near_duplicate_candidates: int = 0
    low_quality_candidates: int = 0
    low_quality_threshold: float = 0.45

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EpisodeConfig":
        values = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        integer_fields = (
            "n_episodes",
            "support_source_groups_per_episode",
            "heldout_source_groups_per_episode",
            "support_clips_per_group",
            "known_like_candidates_per_group",
            "heldout_candidates_per_group",
            "hidden_targets_per_group",
        )
        for field_name in integer_fields:
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"episodes.{field_name} must be positive.")
        for field_name in ("near_duplicate_candidates", "low_quality_candidates"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"episodes.{field_name} must be non-negative.")
        if not 0.0 <= float(self.low_quality_threshold) <= 1.0:
            raise ValueError("episodes.low_quality_threshold must be in [0, 1].")
        if not str(self.split):
            raise ValueError("episodes.split must be non-empty.")


@dataclass(frozen=True)
class ActiveEpisode:
    episode_id: str
    seed: int
    support_clip_ids: tuple[str, ...]
    candidate_clip_ids: tuple[str, ...]
    hidden_target_clip_ids: tuple[str, ...]
    distractor_clip_ids: tuple[str, ...]
    heldout_source_groups: tuple[str, ...]
    known_source_groups: tuple[str, ...]
    candidate_roles: Mapping[str, str]
    low_quality_clip_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "seed": int(self.seed),
            "support_clip_ids": list(self.support_clip_ids),
            "candidate_clip_ids": list(self.candidate_clip_ids),
            "hidden_target_clip_ids": list(self.hidden_target_clip_ids),
            "distractor_clip_ids": list(self.distractor_clip_ids),
            "heldout_source_groups": list(self.heldout_source_groups),
            "known_source_groups": list(self.known_source_groups),
            "candidate_roles": dict(sorted(self.candidate_roles.items())),
            "candidate_role_counts": dict(sorted(Counter(self.candidate_roles.values()).items())),
            "low_quality_clip_ids": list(self.low_quality_clip_ids),
        }


def generate_active_episodes(
    registry: ClipRegistry,
    config: EpisodeConfig | Mapping[str, Any],
) -> list[ActiveEpisode]:
    episode_config = config if isinstance(config, EpisodeConfig) else EpisodeConfig.from_mapping(config)
    episode_config.validate()
    grouped = registry.by_source_group(episode_config.split)
    if not grouped:
        raise EpisodeGenerationError(f"No cached registry clips found for split '{episode_config.split}'.")

    support_needed = int(episode_config.support_clips_per_group) + int(episode_config.known_like_candidates_per_group)
    heldout_needed = int(episode_config.heldout_candidates_per_group) + int(episode_config.hidden_targets_per_group)
    support_eligible = sorted(group for group, rows in grouped.items() if len(rows) >= support_needed)
    heldout_eligible = sorted(group for group, rows in grouped.items() if len(rows) >= heldout_needed)
    if len(support_eligible) < int(episode_config.support_source_groups_per_episode) + int(
        episode_config.heldout_source_groups_per_episode
    ):
        raise EpisodeGenerationError("Not enough source groups with support/candidate capacity for active episodes.")
    if len(heldout_eligible) < int(episode_config.heldout_source_groups_per_episode):
        raise EpisodeGenerationError("Not enough heldout-capable source groups for active episodes.")

    episodes: list[ActiveEpisode] = []
    for episode_index in range(int(episode_config.n_episodes)):
        episode_seed = int(episode_config.seed) + episode_index
        rng = np.random.default_rng(episode_seed)
        heldout_groups = _choose_groups(
            heldout_eligible,
            int(episode_config.heldout_source_groups_per_episode),
            rng,
        )
        known_pool = [group for group in support_eligible if group not in set(heldout_groups)]
        known_groups = _choose_groups(
            known_pool,
            int(episode_config.support_source_groups_per_episode),
            rng,
        )
        episode = _build_one_episode(
            registry=registry,
            grouped=grouped,
            config=episode_config,
            episode_index=episode_index,
            episode_seed=episode_seed,
            known_groups=known_groups,
            heldout_groups=heldout_groups,
            rng=rng,
        )
        episodes.append(episode)
    return episodes


def run_active_episode_smoke(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_active_episode_smoke_config(config)
    mode = "smoke" if smoke else "full"
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_config = EpisodeConfig.from_mapping(config["episodes"])
    if smoke:
        episode_config = replace(episode_config, n_episodes=int(config["execution"].get("smoke_episodes", 1)))

    log_event("active_episode_smoke", "start", mode=mode, n_episodes=episode_config.n_episodes)
    registry = load_clip_registry_from_config(config)
    episodes = generate_active_episodes(registry, episode_config)
    episodes_path = output_dir / f"active_episodes_{mode}.jsonl"
    report_path = output_dir / f"active_episode_report_{mode}.json"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), sort_keys=True) + "\n")

    role_counts: Counter[str] = Counter()
    for episode in episodes:
        role_counts.update(episode.candidate_roles.values())
    report = {
        "mode": mode,
        "registry": {
            "n_clips": len(registry.clips),
            "split_counts": registry.split_counts(),
            "source_group_counts": {
                split: len(registry.source_group_counts(split))
                for split in registry.split_counts()
            },
        },
        "episodes": {
            "n_episodes": len(episodes),
            "candidate_role_counts": dict(sorted(role_counts.items())),
            "support_clip_count_summary": _count_summary(len(episode.support_clip_ids) for episode in episodes),
            "candidate_clip_count_summary": _count_summary(len(episode.candidate_clip_ids) for episode in episodes),
            "hidden_target_clip_count_summary": _count_summary(
                len(episode.hidden_target_clip_ids) for episode in episodes
            ),
        },
        "diagnostics_gate": episode_diagnostics_gate(
            episodes,
            registry,
            config.get("diagnostics", {}),
            low_quality_threshold=float(config["episodes"].get("low_quality_threshold", 0.45)),
            split=episode_config.split,
        ),
        "config": {
            "split": episode_config.split,
            "seed": episode_config.seed,
            "pretrain_manifest": config["data"]["manifests"].get("pretrain", ""),
        },
    }
    if bool(config.get("diagnostics", {}).get("fail_on_violation", False)) and report["diagnostics_gate"]["violations"]:
        raise EpisodeGenerationError(
            "Active episode diagnostics gate failed: "
            + "; ".join(str(violation) for violation in report["diagnostics_gate"]["violations"])
        )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "n_registry_clips": len(registry.clips),
        "n_episodes": len(episodes),
        "report_path": str(report_path),
        "episodes_path": str(episodes_path),
    }
    log_event("active_episode_smoke", "done", **result)
    return result


def validate_active_episode_smoke_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    _required_mapping(config, "episodes")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Active episode smoke must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    if not manifests:
        raise ValueError("data.manifests must not be empty.")
    for split, manifest in manifests.items():
        if not str(split):
            raise ValueError("data.manifests split names must be non-empty.")
        if not str(manifest).startswith("cache/manifests/"):
            raise ValueError("data.manifests paths must be under cache/manifests/.")
    if int(execution.get("smoke_episodes", 1)) <= 0:
        raise ValueError("execution.smoke_episodes must be positive.")
    EpisodeConfig.from_mapping(config["episodes"])


def episode_diagnostics_gate(
    episodes: Sequence[ActiveEpisode],
    registry: ClipRegistry,
    diagnostics_config: object,
    *,
    low_quality_threshold: float,
    split: str,
) -> dict[str, object]:
    config = diagnostics_config if isinstance(diagnostics_config, Mapping) else {}
    required_roles = config.get("required_candidate_roles", ["known_like", "heldout_novel", "near_duplicate"])
    if not isinstance(required_roles, list | tuple):
        raise ValueError("diagnostics.required_candidate_roles must be a list when provided.")
    role_counts: Counter[str] = Counter()
    support_target_violations = 0
    for episode in episodes:
        role_counts.update(episode.candidate_roles.values())
        support_groups = {registry.by_sample_id[sample_id].source_group_id for sample_id in episode.support_clip_ids}
        target_groups = {registry.by_sample_id[sample_id].source_group_id for sample_id in episode.hidden_target_clip_ids}
        if support_groups & target_groups:
            support_target_violations += 1

    low_quality_available = any(
        clip.quality_score is not None and float(clip.quality_score) <= float(low_quality_threshold)
        for clip in registry.clips_for_split(split)
    )
    violations: list[str] = []
    warnings: list[str] = []
    if support_target_violations:
        violations.append(f"support_target_same_group_violations={support_target_violations}")
    for role in required_roles:
        role_name = str(role)
        if role_name == "low_quality" and not low_quality_available:
            warnings.append("low_quality role not required because quality metadata has no low-quality clips.")
            continue
        if role_counts.get(role_name, 0) <= 0:
            violations.append(f"missing_candidate_role:{role_name}")
    return {
        "passed": not violations,
        "violations": violations,
        "warnings": warnings,
        "candidate_role_counts": dict(sorted(role_counts.items())),
        "low_quality_available": bool(low_quality_available),
        "support_target_same_group_violations": int(support_target_violations),
    }


def _build_one_episode(
    *,
    registry: ClipRegistry,
    grouped: Mapping[str, list[ClipRecord]],
    config: EpisodeConfig,
    episode_index: int,
    episode_seed: int,
    known_groups: Sequence[str],
    heldout_groups: Sequence[str],
    rng: np.random.Generator,
) -> ActiveEpisode:
    support_ids: list[str] = []
    candidate_ids: list[str] = []
    hidden_ids: list[str] = []
    candidate_roles: dict[str, str] = {}
    used_ids: set[str] = set()

    for group in known_groups:
        rows = _shuffled_rows(grouped[group], rng)
        known_candidates = rows[: int(config.known_like_candidates_per_group)]
        support_rows = rows[
            int(config.known_like_candidates_per_group) : int(config.known_like_candidates_per_group)
            + int(config.support_clips_per_group)
        ]
        support_ids.extend(row.sample_id for row in support_rows)
        used_ids.update(row.sample_id for row in support_rows)
        _append_candidates(candidate_ids, candidate_roles, used_ids, known_candidates, role="known_like")

    for group in heldout_groups:
        rows = _shuffled_rows(grouped[group], rng)
        heldout_candidates = rows[: int(config.heldout_candidates_per_group)]
        target_rows = rows[
            int(config.heldout_candidates_per_group) : int(config.heldout_candidates_per_group)
            + int(config.hidden_targets_per_group)
        ]
        hidden_ids.extend(row.sample_id for row in target_rows)
        used_ids.update(row.sample_id for row in target_rows)
        _append_candidates(candidate_ids, candidate_roles, used_ids, heldout_candidates, role="heldout_novel")

    near_duplicate_rows = _near_duplicate_rows(
        grouped=grouped,
        candidate_ids=candidate_ids,
        registry=registry,
        used_ids=used_ids,
        max_rows=int(config.near_duplicate_candidates),
        rng=rng,
    )
    _append_candidates(candidate_ids, candidate_roles, used_ids, near_duplicate_rows, role="near_duplicate")

    low_quality_rows = _low_quality_rows(
        registry.clips_for_split(config.split),
        used_ids=used_ids,
        threshold=float(config.low_quality_threshold),
        max_rows=int(config.low_quality_candidates),
    )
    _append_candidates(candidate_ids, candidate_roles, used_ids, low_quality_rows, role="low_quality")

    _validate_episode_ids(
        support_ids=support_ids,
        candidate_ids=candidate_ids,
        hidden_ids=hidden_ids,
        known_groups=known_groups,
        heldout_groups=heldout_groups,
    )
    distractor_ids = tuple(sample_id for sample_id in candidate_ids if candidate_roles[sample_id] != "heldout_novel")
    low_quality_ids = tuple(sample_id for sample_id in candidate_ids if candidate_roles[sample_id] == "low_quality")
    return ActiveEpisode(
        episode_id=f"episode_{episode_index:05d}",
        seed=episode_seed,
        support_clip_ids=tuple(support_ids),
        candidate_clip_ids=tuple(candidate_ids),
        hidden_target_clip_ids=tuple(hidden_ids),
        distractor_clip_ids=distractor_ids,
        heldout_source_groups=tuple(sorted(heldout_groups)),
        known_source_groups=tuple(sorted(known_groups)),
        candidate_roles=candidate_roles,
        low_quality_clip_ids=low_quality_ids,
    )


def _choose_groups(groups: Sequence[str], count: int, rng: np.random.Generator) -> list[str]:
    if len(groups) < count:
        raise EpisodeGenerationError(f"Need {count} source groups, found {len(groups)}.")
    indices = rng.choice(len(groups), size=count, replace=False)
    return [groups[int(idx)] for idx in indices]


def _shuffled_rows(rows: Sequence[ClipRecord], rng: np.random.Generator) -> list[ClipRecord]:
    ordered = sorted(rows, key=lambda row: row.sample_id)
    indices = rng.permutation(len(ordered))
    return [ordered[int(idx)] for idx in indices]


def _append_candidates(
    candidate_ids: list[str],
    candidate_roles: dict[str, str],
    used_ids: set[str],
    rows: Sequence[ClipRecord],
    *,
    role: str,
) -> None:
    for row in rows:
        if row.sample_id in used_ids:
            continue
        candidate_ids.append(row.sample_id)
        candidate_roles[row.sample_id] = role
        used_ids.add(row.sample_id)


def _near_duplicate_rows(
    *,
    grouped: Mapping[str, list[ClipRecord]],
    candidate_ids: Sequence[str],
    registry: ClipRegistry,
    used_ids: set[str],
    max_rows: int,
    rng: np.random.Generator,
) -> list[ClipRecord]:
    if max_rows <= 0:
        return []
    candidate_groups = sorted({registry.by_sample_id[sample_id].source_group_id for sample_id in candidate_ids})
    pool: list[ClipRecord] = []
    for group in candidate_groups:
        pool.extend(row for row in grouped[group] if row.sample_id not in used_ids)
    if not pool:
        return []
    shuffled = _shuffled_rows(pool, rng)
    return shuffled[:max_rows]


def _low_quality_rows(
    rows: Sequence[ClipRecord],
    *,
    used_ids: set[str],
    threshold: float,
    max_rows: int,
) -> list[ClipRecord]:
    if max_rows <= 0:
        return []
    pool = [
        row
        for row in rows
        if row.sample_id not in used_ids
        and row.quality_score is not None
        and float(row.quality_score) <= threshold
    ]
    pool.sort(key=lambda row: (float(row.quality_score), row.source_group_id, row.sample_id))
    return pool[:max_rows]


def _validate_episode_ids(
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    hidden_ids: Sequence[str],
    known_groups: Sequence[str],
    heldout_groups: Sequence[str],
) -> None:
    if set(known_groups) & set(heldout_groups):
        raise EpisodeGenerationError("Known/source support groups overlap heldout groups.")
    groups = [support_ids, candidate_ids, hidden_ids]
    names = ["support", "candidate", "hidden_target"]
    for left_idx, left in enumerate(groups):
        if len(left) != len(set(left)):
            raise EpisodeGenerationError(f"Duplicate clip ids inside {names[left_idx]} set.")
        for right_idx in range(left_idx + 1, len(groups)):
            overlap = set(left) & set(groups[right_idx])
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                raise EpisodeGenerationError(
                    f"{names[left_idx]} and {names[right_idx]} overlap on clip ids: {preview}"
                )
    if not candidate_ids:
        raise EpisodeGenerationError("Episode candidate set is empty.")
    if not hidden_ids:
        raise EpisodeGenerationError("Episode hidden target set is empty.")


def _count_summary(values: Sequence[int] | Any) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(numbers)),
        "mean": float(sum(numbers) / len(numbers)),
        "min": float(min(numbers)),
        "max": float(max(numbers)),
    }


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"active episode config requires object field '{key}'.")
    return value
