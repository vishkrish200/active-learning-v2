from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from marginal_value.active.baselines import (
    BASELINE_POLICY_NAMES,
    blended_kcenter_greedy_quality_gated_order,
    blended_old_novelty_order,
    kcenter_greedy_quality_gated_order,
    old_novelty_only_order,
    quality_only_order,
    random_valid_order,
    window_shape_deterministic_baseline_order,
)
from marginal_value.active.embedding_cache import (
    SUPPORTED_REPRESENTATIONS,
    embedding_cache_dir_from_config,
    load_embedding_lookup,
)
from marginal_value.active.episodes import ActiveEpisode
from marginal_value.active.registry import (
    ClipRecord,
    ClipRegistry,
    audit_clip_registry_coverage_from_config,
    load_clip_registry_from_config,
)
from marginal_value.indexing.cosine_search import mean_nearest_cosine_distance
from marginal_value.indexing.knn_features import normalize_rows
from marginal_value.logging_utils import log_event


ORACLE_POLICY_NAME = "oracle_greedy_eval_only"
BALANCED_REPRESENTATION_NAME = "balanced"
DEFAULT_REPRESENTATIONS = (
    "window_mean_std_pool",
    "temporal_order",
    "raw_shape_stats",
    "window_shape_stats",
)
BLEND_POLICY_REPRESENTATIONS = ("ts2vec", "window_mean_std_pool")


def run_active_loop_eval(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_active_loop_eval_config(config)
    mode = "smoke" if smoke else "full"
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("active_loop_eval", "start", mode=mode)

    registry = load_clip_registry_from_config(config)
    registry_coverage = audit_clip_registry_coverage_from_config(config, registry=registry)
    episodes = load_active_episodes(_episode_path(config))
    if smoke:
        episodes = episodes[: int(config["execution"].get("smoke_max_episodes", 1))]
    if not episodes:
        raise ValueError("Active-loop evaluation requires at least one episode.")

    eval_config = config["evaluation"]
    representations = [str(rep) for rep in eval_config.get("representations", DEFAULT_REPRESENTATIONS)]
    primary_representation = str(eval_config.get("primary_representation", representations[0]))
    if primary_representation not in representations:
        raise ValueError("evaluation.primary_representation must be included in evaluation.representations.")
    k_values = [int(k) for k in eval_config.get("k_values", [10, 25, 50, 100, 200, 400])]
    policies = [str(policy) for policy in eval_config.get("policies", [*BASELINE_POLICY_NAMES, ORACLE_POLICY_NAME])]
    quality_threshold = float(eval_config.get("quality_threshold", 0.45))

    used_sample_ids = _episode_sample_ids(episodes)
    missing_ids = sorted(sample_id for sample_id in used_sample_ids if sample_id not in registry.by_sample_id)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise KeyError(f"Episodes reference {len(missing_ids)} clips absent from the active registry: {preview}")
    embedding_result = load_embedding_lookup(
        [registry.by_sample_id[sample_id] for sample_id in sorted(used_sample_ids)],
        representations=representations,
        sample_rate=float(eval_config.get("sample_rate", 30.0)),
        raw_shape_max_samples=eval_config.get("raw_shape_max_samples"),
        cache_dir=embedding_cache_dir_from_config(config),
        component="active_loop_eval",
        mode=mode,
        representation_options=_representation_options(config, eval_config),
    )
    embedding_lookup = embedding_result.embeddings

    coverage_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    episode_diagnostics = _empty_episode_diagnostics()
    progress_every = max(1, len(episodes) // 10)
    for episode_offset, episode in enumerate(episodes):
        episode_index = episode_offset + 1
        diagnostics = _episode_diagnostics(episode, registry)
        _merge_episode_diagnostics(episode_diagnostics, diagnostics)
        policy_orders, candidate_rows = _policy_orders_for_episode(
            episode=episode,
            registry=registry,
            embeddings=embedding_lookup,
            policies=policies,
            primary_representation=primary_representation,
            quality_threshold=quality_threshold,
            random_seed=int(eval_config.get("random_seed", 17)) + episode_offset,
            old_novelty_k=int(eval_config.get("old_novelty_k", 10)),
            cluster_similarity_threshold=float(eval_config.get("cluster_similarity_threshold", 0.995)),
            max_selected=max(k_values),
        )
        log_event(
            "active_loop_eval",
            "episode_eval_start",
            index=episode_index,
            total=len(episodes),
            every=progress_every,
            mode=mode,
            episode_id=episode.episode_id,
        )
        for policy_name, order in policy_orders.items():
            for k in k_values:
                selected = list(order[: min(int(k), len(order))])
                rep_metrics = _coverage_by_representation(
                    episode=episode,
                    embeddings=embedding_lookup,
                    representations=representations,
                    selected_indices=selected,
                )
                balanced = _balanced_metrics(rep_metrics.values())
                for representation, metrics in [*rep_metrics.items(), (BALANCED_REPRESENTATION_NAME, balanced)]:
                    coverage_rows.append(
                        {
                            "episode_id": episode.episode_id,
                            "policy": policy_name,
                            "k": int(k),
                            "representation": representation,
                            **metrics,
                        }
                    )
                selection_rows.append(
                    _selection_audit_row(
                        episode=episode,
                        registry=registry,
                        candidate_rows=candidate_rows,
                        selected_indices=selected,
                        policy_name=policy_name,
                        k=int(k),
                        quality_threshold=quality_threshold,
                    )
                )
        if episode_index == len(episodes) or episode_index % progress_every == 0:
            log_event(
                "active_loop_eval",
                "episode_eval_progress",
                index=episode_index,
                total=len(episodes),
                mode=mode,
                coverage_rows=len(coverage_rows),
                selection_rows=len(selection_rows),
            )

    coverage_path = output_dir / f"coverage_gain_by_episode_{mode}.csv"
    selection_path = output_dir / f"topk_selection_audit_{mode}.csv"
    report_path = output_dir / f"coverage_gain_report_{mode}.json"
    _add_oracle_fractions(coverage_rows)
    _write_csv(coverage_path, coverage_rows)
    _write_csv(selection_path, selection_rows)
    report = {
        "mode": mode,
        "n_episodes": len(episodes),
        "representations": representations,
        "primary_representation": primary_representation,
        "k_values": k_values,
        "registry_coverage": _deduplicate_coverage_aliases(registry_coverage),
        "registry_coverage_summary": _registry_coverage_summary(registry_coverage),
        "embedding_cache": embedding_result.report(),
        "episode_diagnostics": _finalize_episode_diagnostics(episode_diagnostics),
        "policies": _policy_summary(coverage_rows),
        "artifacts": {
            "coverage_gain_by_episode": str(coverage_path),
            "topk_selection_audit": str(selection_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "n_episodes": len(episodes),
        "report_path": str(report_path),
        "coverage_by_episode_path": str(coverage_path),
        "selection_audit_path": str(selection_path),
    }
    log_event("active_loop_eval", "done", **result)
    return result


def validate_active_loop_eval_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    episodes = _required_mapping(config, "episodes")
    evaluation = _required_mapping(config, "evaluation")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Active-loop evaluation must run on Modal.")
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
    representations = evaluation.get("representations", DEFAULT_REPRESENTATIONS)
    if not isinstance(representations, list | tuple) or not representations:
        raise ValueError("evaluation.representations must be a non-empty list.")
    unsupported = set(str(rep) for rep in representations) - set(SUPPORTED_REPRESENTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported active-loop representations: {sorted(unsupported)}")
    for k in evaluation.get("k_values", [10, 25, 50, 100, 200, 400]):
        if int(k) <= 0:
            raise ValueError("evaluation.k_values must be positive.")
    policies = [str(policy) for policy in evaluation.get("policies", [*BASELINE_POLICY_NAMES, ORACLE_POLICY_NAME])]
    supported = {*BASELINE_POLICY_NAMES, ORACLE_POLICY_NAME}
    unknown = {policy for policy in policies if policy not in supported and _parse_blend_policy(policy) is None}
    if unknown:
        raise ValueError(f"Unsupported active-loop policies: {sorted(unknown)}")
    blend_reps = set(str(rep) for rep in representations)
    blend_missing = sorted(
        {
            rep
            for policy in policies
            for rep in (_parse_blend_policy(policy) or {}).get("representations", ())
            if rep not in blend_reps
        }
    )
    if blend_missing:
        raise ValueError(f"Blended active-loop policies require representations: {blend_missing}")


def load_active_episodes(path: str | Path) -> list[ActiveEpisode]:
    episodes: list[ActiveEpisode] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episodes.append(_episode_from_mapping(json.loads(line)))
    return episodes


def _policy_orders_for_episode(
    *,
    episode: ActiveEpisode,
    registry: ClipRegistry,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    policies: Sequence[str],
    primary_representation: str,
    quality_threshold: float,
    random_seed: int,
    old_novelty_k: int,
    cluster_similarity_threshold: float,
    max_selected: int,
) -> tuple[dict[str, list[int]], list[dict[str, object]]]:
    support_ids = list(episode.support_clip_ids)
    candidate_ids = list(episode.candidate_clip_ids)
    primary = embeddings[primary_representation]
    support_embeddings = _stack(primary, support_ids)
    candidate_embeddings = _stack(primary, candidate_ids)
    old_order, old_novelty = old_novelty_only_order(
        support_embeddings,
        candidate_embeddings,
        k=min(max(1, int(old_novelty_k)), len(support_ids)),
    )
    candidate_rows = _candidate_rows(
        episode=episode,
        registry=registry,
        candidate_ids=candidate_ids,
        old_novelty=old_novelty,
        cluster_ids=_candidate_clusters(candidate_embeddings, similarity_threshold=cluster_similarity_threshold),
    )

    orders: dict[str, list[int]] = {}
    for policy in policies:
        blend_policy = _parse_blend_policy(policy)
        if blend_policy is not None:
            left_rep, right_rep = blend_policy["representations"]
            if left_rep not in embeddings or right_rep not in embeddings:
                raise ValueError(f"Blended policy {policy} requires {left_rep!r} and {right_rep!r} embeddings.")
            left_support = _stack(embeddings[left_rep], support_ids)
            left_candidates = _stack(embeddings[left_rep], candidate_ids)
            right_support = _stack(embeddings[right_rep], support_ids)
            right_candidates = _stack(embeddings[right_rep], candidate_ids)
            if blend_policy["kind"] == "old_novelty":
                _left_order, left_novelty = old_novelty_only_order(
                    left_support,
                    left_candidates,
                    k=min(max(1, int(old_novelty_k)), len(support_ids)),
                )
                _right_order, right_novelty = old_novelty_only_order(
                    right_support,
                    right_candidates,
                    k=min(max(1, int(old_novelty_k)), len(support_ids)),
                )
                orders[policy] = blended_old_novelty_order(
                    left_novelty,
                    right_novelty,
                    alpha=float(blend_policy["alpha"]),
                )
            elif blend_policy["kind"] == "kcenter":
                orders[policy] = blended_kcenter_greedy_quality_gated_order(
                    left_support,
                    left_candidates,
                    right_support,
                    right_candidates,
                    candidate_rows,
                    alpha=float(blend_policy["alpha"]),
                    quality_threshold=quality_threshold,
                )
            else:
                raise ValueError(f"Unsupported blended policy kind: {blend_policy['kind']}")
        elif policy == "random_valid":
            orders[policy] = random_valid_order(candidate_rows, seed=random_seed, quality_threshold=quality_threshold)
        elif policy == "quality_only":
            orders[policy] = quality_only_order(candidate_rows)
        elif policy == "old_novelty_only":
            orders[policy] = old_order
        elif policy == "kcenter_greedy_quality_gated":
            orders[policy] = kcenter_greedy_quality_gated_order(
                support_embeddings,
                candidate_embeddings,
                candidate_rows,
                quality_threshold=quality_threshold,
            )
        elif policy == "window_shape_stats_q85_stat90_abs60_clustercap2":
            rows = candidate_rows
            if "window_shape_stats" in embeddings:
                shape_order, shape_novelty = old_novelty_only_order(
                    _stack(embeddings["window_shape_stats"], support_ids),
                    _stack(embeddings["window_shape_stats"], candidate_ids),
                    k=min(max(1, int(old_novelty_k)), len(support_ids)),
                )
                del shape_order
                rows = [dict(row, old_novelty_score=float(shape_novelty[idx])) for idx, row in enumerate(candidate_rows)]
            orders[policy] = window_shape_deterministic_baseline_order(
                rows,
                quality_threshold=0.85,
                max_stationary_fraction=0.90,
                max_abs_value=60.0,
                cluster_cap=2,
            )
        elif policy == ORACLE_POLICY_NAME:
            orders[policy] = _oracle_greedy_order(
                episode=episode,
                embeddings=embeddings,
                representations=list(embeddings),
                max_selected=max_selected,
            )
        else:
            raise ValueError(f"Unsupported policy: {policy}")
    return orders, candidate_rows


def _coverage_by_representation(
    *,
    episode: ActiveEpisode,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    representations: Sequence[str],
    selected_indices: Sequence[int],
) -> dict[str, dict[str, float]]:
    candidate_ids = list(episode.candidate_clip_ids)
    selected_ids = [candidate_ids[int(idx)] for idx in selected_indices]
    metrics: dict[str, dict[str, float]] = {}
    for representation in representations:
        lookup = embeddings[representation]
        metrics[representation] = coverage_gain_for_ids(
            support_clip_ids=episode.support_clip_ids,
            hidden_target_clip_ids=episode.hidden_target_clip_ids,
            selected_clip_ids=selected_ids,
            embeddings=lookup,
        )
    return metrics


def coverage_gain_for_ids(
    *,
    support_clip_ids: Sequence[str],
    hidden_target_clip_ids: Sequence[str],
    selected_clip_ids: Sequence[str],
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, float]:
    support = _stack(embeddings, support_clip_ids)
    target = _stack(embeddings, hidden_target_clip_ids)
    before = mean_nearest_cosine_distance(target, support, backend="auto")
    if selected_clip_ids:
        augmented = np.vstack([support, _stack(embeddings, selected_clip_ids)])
    else:
        augmented = support
    after = mean_nearest_cosine_distance(target, augmented, backend="auto")
    gain = float(before - after)
    return {
        "coverage_before": float(before),
        "coverage_after": float(after),
        "absolute_gain": gain,
        "relative_gain": float(gain / before) if before > 1.0e-12 else 0.0,
        "selected_count": float(len(selected_clip_ids)),
    }


def _oracle_greedy_order(
    *,
    episode: ActiveEpisode,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    representations: Sequence[str],
    max_selected: int,
) -> list[int]:
    candidate_ids = list(episode.candidate_clip_ids)
    if not candidate_ids:
        return []
    remaining = np.ones(len(candidate_ids), dtype=bool)
    ordered: list[int] = []
    state = _build_oracle_state(
        episode=episode,
        embeddings=embeddings,
        representations=representations,
    )
    selected_limit = min(max(0, int(max_selected)), len(candidate_ids))
    while len(ordered) < selected_limit and np.any(remaining):
        scores = np.zeros(len(candidate_ids), dtype=float)
        for rep_state in state:
            current = rep_state["current_distances"]
            candidate_distances = rep_state["candidate_distances"]
            after = np.minimum(current[:, None], candidate_distances)
            after_mean = np.mean(after, axis=0)
            baseline = float(rep_state["baseline_mean"])
            if baseline > 1.0e-12:
                scores += (baseline - after_mean) / baseline
            else:
                scores += baseline - after_mean
        scores /= max(1, len(state))
        scores[~remaining] = -np.inf
        best_score = float(np.max(scores))
        if not np.isfinite(best_score):
            break
        tied = np.flatnonzero(np.isclose(scores, best_score, rtol=1.0e-12, atol=1.0e-12))
        best_idx = int(sorted(tied.tolist(), key=lambda idx: candidate_ids[int(idx)])[0])
        ordered.append(best_idx)
        remaining[best_idx] = False
        for rep_state in state:
            rep_state["current_distances"] = np.minimum(
                rep_state["current_distances"],
                rep_state["candidate_distances"][:, best_idx],
            )
    if len(ordered) < len(candidate_ids):
        ordered_set = set(ordered)
        ordered.extend(idx for idx in range(len(candidate_ids)) if idx not in ordered_set)
    return ordered


def _build_oracle_state(
    *,
    episode: ActiveEpisode,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    representations: Sequence[str],
) -> list[dict[str, np.ndarray | float]]:
    state: list[dict[str, np.ndarray | float]] = []
    candidate_ids = list(episode.candidate_clip_ids)
    for representation in representations:
        lookup = embeddings[representation]
        support = normalize_rows(_stack(lookup, episode.support_clip_ids))
        target = normalize_rows(_stack(lookup, episode.hidden_target_clip_ids))
        candidates = normalize_rows(_stack(lookup, candidate_ids))
        support_distances = 1.0 - target @ support.T
        current = np.min(support_distances, axis=1)
        candidate_distances = 1.0 - target @ candidates.T
        state.append(
            {
                "baseline_mean": float(np.mean(current)),
                "current_distances": current,
                "candidate_distances": candidate_distances,
            }
        )
    return state


def _candidate_rows(
    *,
    episode: ActiveEpisode,
    registry: ClipRegistry,
    candidate_ids: Sequence[str],
    old_novelty: np.ndarray,
    cluster_ids: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    novelty_scores = _minmax(old_novelty)
    for idx, sample_id in enumerate(candidate_ids):
        clip = registry.by_sample_id[str(sample_id)]
        role = str(episode.candidate_roles.get(str(sample_id), "unlabeled"))
        row: dict[str, object] = {
            "candidate_index": int(idx),
            "sample_id": clip.sample_id,
            "worker_id": clip.worker_id,
            "source_group_id": clip.source_group_id,
            "candidate_role": role,
            "quality_score": _quality_value(clip, "quality_score", default=1.0),
            "stationary_fraction": _quality_value(clip, "stationary_fraction", default=0.0),
            "max_abs_value": _quality_value(clip, "max_abs_value", default=0.0),
            "old_knn_distance": float(old_novelty[idx]),
            "old_novelty_score": float(novelty_scores[idx]),
            "new_cluster_id": int(cluster_ids[idx]),
        }
        rows.append(row)
    return rows


def _selection_audit_row(
    *,
    episode: ActiveEpisode,
    registry: ClipRegistry,
    candidate_rows: Sequence[Mapping[str, object]],
    selected_indices: Sequence[int],
    policy_name: str,
    k: int,
    quality_threshold: float,
) -> dict[str, object]:
    selected_rows = [candidate_rows[int(idx)] for idx in selected_indices]
    selected_count = len(selected_rows)
    role_counts = Counter(str(row.get("candidate_role", "unlabeled")) for row in selected_rows)
    source_groups = {str(row.get("source_group_id", "")) for row in selected_rows}
    clusters = {str(row.get("new_cluster_id", "")) for row in selected_rows}
    low_quality_count = sum(_safe_float(row.get("quality_score", 1.0)) < quality_threshold for row in selected_rows)
    artifact_count = sum(
        _safe_float(row.get("quality_score", 1.0)) < quality_threshold
        or str(row.get("candidate_role", "")) == "low_quality"
        for row in selected_rows
    )
    target_groups = {registry.by_sample_id[sample_id].source_group_id for sample_id in episode.hidden_target_clip_ids}
    support_groups = {registry.by_sample_id[sample_id].source_group_id for sample_id in episode.support_clip_ids}
    same_target = sum(str(row.get("source_group_id", "")) in target_groups for row in selected_rows)
    same_support = sum(str(row.get("source_group_id", "")) in support_groups for row in selected_rows)
    return {
        "episode_id": episode.episode_id,
        "policy": policy_name,
        "k": int(k),
        "selected_count": int(selected_count),
        "artifact_rate_at_k": _fraction(artifact_count, selected_count),
        "low_quality_rate_at_k": _fraction(low_quality_count, selected_count),
        "duplicate_rate_at_k": 1.0 - _fraction(len(clusters), selected_count),
        "candidate_role_mix_json": json.dumps(dict(sorted(role_counts.items())), sort_keys=True),
        "unique_source_groups_at_k": int(len(source_groups)),
        "unique_new_clusters_at_k": int(len(clusters)),
        "candidate_target_same_group_rate_at_k": _fraction(same_target, selected_count),
        "candidate_support_same_group_rate_at_k": _fraction(same_support, selected_count),
    }


def _episode_diagnostics(episode: ActiveEpisode, registry: ClipRegistry) -> dict[str, object]:
    support_groups = {registry.by_sample_id[sample_id].source_group_id for sample_id in episode.support_clip_ids}
    target_groups = {registry.by_sample_id[sample_id].source_group_id for sample_id in episode.hidden_target_clip_ids}
    candidate_groups = [registry.by_sample_id[sample_id].source_group_id for sample_id in episode.candidate_clip_ids]
    role_counts = Counter(str(episode.candidate_roles.get(sample_id, "unlabeled")) for sample_id in episode.candidate_clip_ids)
    return {
        "episode_count": 1,
        "support_clip_count": len(episode.support_clip_ids),
        "candidate_clip_count": len(episode.candidate_clip_ids),
        "hidden_target_clip_count": len(episode.hidden_target_clip_ids),
        "heldout_group_count": len(episode.heldout_source_groups),
        "known_group_count": len(episode.known_source_groups),
        "low_quality_count": len(episode.low_quality_clip_ids),
        "near_duplicate_count": role_counts.get("near_duplicate", 0),
        "support_target_same_group_violations": int(bool(support_groups & target_groups)),
        "candidate_target_same_group_count": sum(group in target_groups for group in candidate_groups),
        "candidate_support_same_group_count": sum(group in support_groups for group in candidate_groups),
        "candidate_count_for_overlap": len(candidate_groups),
        "candidate_role_counts": dict(role_counts),
    }


def _empty_episode_diagnostics() -> dict[str, object]:
    return {
        "episode_count": 0,
        "support_clip_count": [],
        "candidate_clip_count": [],
        "hidden_target_clip_count": [],
        "heldout_group_count": [],
        "known_group_count": [],
        "low_quality_count": [],
        "near_duplicate_count": [],
        "support_target_same_group_violations": 0,
        "candidate_target_same_group_count": 0,
        "candidate_support_same_group_count": 0,
        "candidate_count_for_overlap": 0,
        "candidate_role_counts": Counter(),
    }


def _merge_episode_diagnostics(total: dict[str, object], row: Mapping[str, object]) -> None:
    total["episode_count"] = int(total["episode_count"]) + 1
    for key in (
        "support_clip_count",
        "candidate_clip_count",
        "hidden_target_clip_count",
        "heldout_group_count",
        "known_group_count",
        "low_quality_count",
        "near_duplicate_count",
    ):
        values = total[key]
        if isinstance(values, list):
            values.append(int(row[key]))
    for key in (
        "support_target_same_group_violations",
        "candidate_target_same_group_count",
        "candidate_support_same_group_count",
        "candidate_count_for_overlap",
    ):
        total[key] = int(total[key]) + int(row[key])
    role_counts = total["candidate_role_counts"]
    if isinstance(role_counts, Counter):
        role_counts.update(row.get("candidate_role_counts", {}))


def _finalize_episode_diagnostics(total: Mapping[str, object]) -> dict[str, object]:
    candidate_count = int(total["candidate_count_for_overlap"])
    role_counts = total["candidate_role_counts"]
    return {
        "episode_count": int(total["episode_count"]),
        "support_clip_count": _numeric_summary(total["support_clip_count"]),
        "candidate_clip_count": _numeric_summary(total["candidate_clip_count"]),
        "hidden_target_clip_count": _numeric_summary(total["hidden_target_clip_count"]),
        "heldout_group_count": _numeric_summary(total["heldout_group_count"]),
        "known_group_count": _numeric_summary(total["known_group_count"]),
        "low_quality_count": _numeric_summary(total["low_quality_count"]),
        "near_duplicate_count": _numeric_summary(total["near_duplicate_count"]),
        "support_target_same_group_violations": int(total["support_target_same_group_violations"]),
        "candidate_target_same_group_rate": _fraction(int(total["candidate_target_same_group_count"]), candidate_count),
        "candidate_support_same_group_rate": _fraction(int(total["candidate_support_same_group_count"]), candidate_count),
        "candidate_role_counts": dict(sorted(role_counts.items())) if isinstance(role_counts, Counter) else {},
    }


def _candidate_clusters(embeddings: np.ndarray, *, similarity_threshold: float) -> np.ndarray:
    vectors = normalize_rows(np.asarray(embeddings, dtype=float))
    n_rows = len(vectors)
    if n_rows == 0:
        return np.asarray([], dtype=int)
    similarities = vectors @ vectors.T
    visited = np.zeros(n_rows, dtype=bool)
    cluster_ids = np.full(n_rows, -1, dtype=int)
    cluster_id = 0
    for start in range(n_rows):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        cluster_ids[start] = cluster_id
        while stack:
            idx = stack.pop()
            neighbors = np.flatnonzero((similarities[idx] >= similarity_threshold) & ~visited)
            for neighbor in neighbors:
                visited[neighbor] = True
                cluster_ids[neighbor] = cluster_id
                stack.append(int(neighbor))
        cluster_id += 1
    return cluster_ids


def _policy_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    grouped: defaultdict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), f"coverage@{int(row['k'])}", str(row["representation"]))].append(row)
    summary: dict[str, dict[str, object]] = defaultdict(dict)
    for (policy, k_key, representation), values in sorted(grouped.items()):
        summary.setdefault(policy, {}).setdefault(k_key, {})[representation] = {
            "coverage_before_mean": _mean(row["coverage_before"] for row in values),
            "coverage_after_mean": _mean(row["coverage_after"] for row in values),
            "absolute_gain_mean": _mean(row["absolute_gain"] for row in values),
            "relative_gain_mean": _mean(row["relative_gain"] for row in values),
            "oracle_fraction_mean": _mean(row.get("oracle_fraction", 0.0) for row in values),
        }
    return {policy: dict(policy_summary) for policy, policy_summary in sorted(summary.items())}


def _add_oracle_fractions(rows: Sequence[dict[str, object]]) -> None:
    oracle_gain_by_key: dict[tuple[str, int, str], float] = {}
    for row in rows:
        if str(row.get("policy", "")) != ORACLE_POLICY_NAME:
            continue
        key = (str(row["episode_id"]), int(row["k"]), str(row["representation"]))
        oracle_gain_by_key[key] = float(row.get("absolute_gain", 0.0))
    for row in rows:
        key = (str(row["episode_id"]), int(row["k"]), str(row["representation"]))
        oracle_gain = oracle_gain_by_key.get(key, 0.0)
        row["oracle_absolute_gain"] = float(oracle_gain)
        row["oracle_fraction"] = float(float(row.get("absolute_gain", 0.0)) / oracle_gain) if oracle_gain > 1.0e-12 else 0.0


def _balanced_metrics(metrics: Iterable[Mapping[str, float]]) -> dict[str, float]:
    rows = list(metrics)
    return {
        "coverage_before": _mean(row["coverage_before"] for row in rows),
        "coverage_after": _mean(row["coverage_after"] for row in rows),
        "absolute_gain": _mean(row["absolute_gain"] for row in rows),
        "relative_gain": _mean(row["relative_gain"] for row in rows),
        "selected_count": _mean(row["selected_count"] for row in rows),
    }


def _episode_from_mapping(row: Mapping[str, Any]) -> ActiveEpisode:
    return ActiveEpisode(
        episode_id=str(row["episode_id"]),
        seed=int(row.get("seed", 0)),
        support_clip_ids=tuple(str(value) for value in row.get("support_clip_ids", [])),
        candidate_clip_ids=tuple(str(value) for value in row.get("candidate_clip_ids", [])),
        hidden_target_clip_ids=tuple(str(value) for value in row.get("hidden_target_clip_ids", [])),
        distractor_clip_ids=tuple(str(value) for value in row.get("distractor_clip_ids", [])),
        heldout_source_groups=tuple(str(value) for value in row.get("heldout_source_groups", [])),
        known_source_groups=tuple(str(value) for value in row.get("known_source_groups", [])),
        candidate_roles={str(key): str(value) for key, value in row.get("candidate_roles", {}).items()},
        low_quality_clip_ids=tuple(str(value) for value in row.get("low_quality_clip_ids", [])),
    )


def _episode_path(config: Mapping[str, Any]) -> Path:
    path = Path(str(config["episodes"]["path"]))
    if path.is_absolute():
        return path
    return Path(str(config["data"]["root"])) / path


def _episode_sample_ids(episodes: Sequence[ActiveEpisode]) -> set[str]:
    sample_ids: set[str] = set()
    for episode in episodes:
        sample_ids.update(episode.support_clip_ids)
        sample_ids.update(episode.candidate_clip_ids)
        sample_ids.update(episode.hidden_target_clip_ids)
    return sample_ids


def _stack(embeddings: Mapping[str, np.ndarray], sample_ids: Sequence[str]) -> np.ndarray:
    if not sample_ids:
        raise ValueError("Cannot stack embeddings for an empty sample-id list.")
    return np.vstack([embeddings[str(sample_id)] for sample_id in sample_ids])


def _quality_value(clip: ClipRecord, key: str, *, default: float) -> float:
    try:
        return float(clip.quality.get(key, default))
    except (TypeError, ValueError):
        return default


def _minmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    lo = float(np.min(array))
    hi = float(np.max(array))
    if hi - lo < 1.0e-12:
        return np.zeros_like(array, dtype=float)
    return (array - lo) / (hi - lo)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[object]) -> float:
    numbers = [float(value) for value in values]
    return float(np.mean(numbers)) if numbers else 0.0


def _numeric_summary(values: object) -> dict[str, float]:
    numbers = [float(value) for value in values] if isinstance(values, list) else []
    if not numbers:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(numbers)),
        "mean": float(np.mean(numbers)),
        "min": float(min(numbers)),
        "max": float(max(numbers)),
    }


def _deduplicate_coverage_aliases(coverage: Mapping[str, Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    seen_ids: set[int] = set()
    for key, value in coverage.items():
        if key in {"old", "new"}:
            output[key] = value
            seen_ids.add(id(value))
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


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"active-loop eval config requires object field '{key}'.")
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


def _parse_blend_policy(policy: str) -> dict[str, object] | None:
    prefixes = {
        "blend_old_novelty_ts2vec_window_mean_std_pool_a": "old_novelty",
        "blend_kcenter_ts2vec_window_mean_std_pool_a": "kcenter",
    }
    for prefix, kind in prefixes.items():
        if not policy.startswith(prefix):
            continue
        suffix = policy[len(prefix):]
        if not suffix.isdigit():
            return None
        alpha_tenths = int(suffix)
        if alpha_tenths < 0 or alpha_tenths > 10:
            return None
        return {
            "kind": kind,
            "alpha": float(alpha_tenths / 10.0),
            "representations": BLEND_POLICY_REPRESENTATIONS,
        }
    return None
