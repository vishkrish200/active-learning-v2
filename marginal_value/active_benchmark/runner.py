from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from marginal_value.active_benchmark.metrics import (
    balanced_relative_gain,
    coverage_report_for_selection,
    selection_summary,
    summarize_policy_rounds,
)
from marginal_value.active_benchmark.policies import select_batch
from marginal_value.active_benchmark.representations import clip_lookup, stack_embeddings
from marginal_value.active_benchmark.schema import BenchmarkClip, BenchmarkResult, EpisodeSpec, OfflineBenchmarkConfig, RoundResult
from marginal_value.indexing.knn_features import normalize_rows


def run_offline_active_benchmark(
    clips: Sequence[BenchmarkClip],
    episodes: Sequence[EpisodeSpec],
    config: OfflineBenchmarkConfig,
    *,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> BenchmarkResult:
    if int(config.batch_size) <= 0:
        raise ValueError("Offline active benchmark batch_size must be positive.")
    if int(config.rounds) <= 0:
        raise ValueError("Offline active benchmark rounds must be positive.")
    clips_by_id = clip_lookup(clips)
    _validate_config(config)
    _validate_episodes(episodes, clips_by_id)

    round_results: list[RoundResult] = []
    for episode_index, episode in enumerate(episodes):
        _emit_progress(
            progress_callback,
            "episode_start",
            episode_id=episode.episode_id,
            fold_id=int(episode.fold_id),
            support_count=len(episode.support_ids),
            candidate_count=len(episode.candidate_ids),
            target_count=len(episode.target_ids),
        )
        for policy_name in config.policies:
            _emit_progress(
                progress_callback,
                "policy_start",
                episode_id=episode.episode_id,
                fold_id=int(episode.fold_id),
                policy_name=str(policy_name),
            )
            support_ids = list(episode.support_ids)
            candidate_ids = list(episode.candidate_ids)
            acquired_ids: list[str] = []
            for round_index in range(int(config.rounds)):
                support_before = tuple(support_ids)
                candidate_before = tuple(candidate_ids)
                selected_ids, selected_scores = select_batch(
                    str(policy_name),
                    clips_by_id,
                    support_ids=support_before,
                    candidate_ids=candidate_before,
                    target_ids=episode.target_ids,
                    config=config,
                    episode_index=episode_index,
                    round_index=round_index,
                )
                coverage = coverage_report_for_selection(
                    clips_by_id,
                    support_ids=support_before,
                    target_ids=episode.target_ids,
                    candidate_ids=candidate_before,
                    selected_ids=selected_ids,
                    representations=config.representations,
                )
                balanced_gain = balanced_relative_gain(
                    coverage,
                    primary_representations=config.primary_representations,
                )
                selected_set = set(selected_ids)
                support_ids = [*support_ids, *selected_ids]
                acquired_ids.extend(selected_ids)
                candidate_ids = [sample_id for sample_id in candidate_ids if sample_id not in selected_set]
                cumulative_coverage = coverage_report_for_selection(
                    clips_by_id,
                    support_ids=episode.support_ids,
                    target_ids=episode.target_ids,
                    candidate_ids=episode.candidate_ids,
                    selected_ids=acquired_ids,
                    representations=config.representations,
                )
                cumulative_balanced_gain = balanced_relative_gain(
                    cumulative_coverage,
                    primary_representations=config.primary_representations,
                )
                round_result = RoundResult(
                    episode_id=episode.episode_id,
                    fold_id=episode.fold_id,
                    policy_name=str(policy_name),
                    round_index=int(round_index),
                    batch_size=int(config.batch_size),
                    support_ids_before=support_before,
                    candidate_ids_before=candidate_before,
                    selected_ids=tuple(selected_ids),
                    selected_scores=tuple(selected_scores),
                    selected_source_group_ids=tuple(clips_by_id[sample_id].source_group_id for sample_id in selected_ids),
                    support_ids_after=tuple(support_ids),
                    candidate_ids_after=tuple(candidate_ids),
                    coverage_by_representation=coverage,
                    balanced_relative_gain=balanced_gain,
                    cumulative_balanced_relative_gain=float(cumulative_balanced_gain),
                    selection_summary=selection_summary(clips_by_id, selected_ids),
                    selection_details=_selection_details(
                        clips_by_id,
                        support_ids=support_before,
                        candidate_ids=candidate_before,
                        selected_ids=selected_ids,
                        selected_scores=selected_scores,
                        config=config,
                    ),
                )
                round_results.append(round_result)
                _emit_progress(
                    progress_callback,
                    "round_done",
                    episode_id=episode.episode_id,
                    fold_id=int(episode.fold_id),
                    policy_name=str(policy_name),
                    round_index=int(round_index),
                    selected_count=len(selected_ids),
                    support_count_before=round_result.support_count_before,
                    support_count_after=round_result.support_count_after,
                    candidate_count_before=round_result.candidate_count_before,
                    candidate_count_after=round_result.candidate_count_after,
                    balanced_relative_gain=float(balanced_gain),
                    cumulative_balanced_relative_gain=float(cumulative_balanced_gain),
                    oracle_candidate_cap=config.oracle_candidate_cap,
                )
    _attach_oracle_fractions(round_results, episodes=episodes, clips_by_id=clips_by_id, config=config)
    result = BenchmarkResult(
        episodes=tuple(episodes),
        policies=tuple(str(policy) for policy in config.policies),
        rounds=tuple(round_results),
        policy_summary=summarize_policy_rounds(round_results),
        difficulty_audit=_summarize_episode_difficulty(
            round_results,
            episodes=episodes,
            clips_by_id=clips_by_id,
            config=config,
        ),
        config=config,
    )
    _emit_progress(
        progress_callback,
        "benchmark_done",
        n_episodes=len(result.episodes),
        n_round_rows=len(result.rounds),
        policies=list(result.policies),
    )
    return result


def _validate_config(config: OfflineBenchmarkConfig) -> None:
    if not config.policies:
        raise ValueError("Offline active benchmark requires at least one policy.")
    if not config.representations:
        raise ValueError("Offline active benchmark requires at least one representation.")
    if not config.primary_representations:
        raise ValueError("Offline active benchmark requires at least one primary representation.")
    missing = set(config.primary_representations) - set(config.representations)
    if missing:
        raise ValueError(f"Primary representations are not loaded: {sorted(missing)}")
    if any(str(policy) in {"blend_kcenter_ts2vec_window", "artifact_gate_blend_kcenter_ts2vec_window"} for policy in config.policies):
        blend_missing = {str(config.blend_left_representation), str(config.blend_right_representation)} - set(config.representations)
        if blend_missing:
            raise ValueError(f"Blend policy representations are not loaded: {sorted(blend_missing)}")
        if not 0.0 <= float(config.blend_alpha) <= 1.0:
            raise ValueError("blend_alpha must be in [0, 1].")
        if config.max_artifact_score is not None and float(config.max_artifact_score) < 0.0:
            raise ValueError("max_artifact_score must be non-negative when provided.")


def _validate_episodes(episodes: Sequence[EpisodeSpec], clips_by_id: dict[str, BenchmarkClip]) -> None:
    if not episodes:
        raise ValueError("Offline active benchmark requires at least one episode.")
    known_ids = set(clips_by_id)
    for episode in episodes:
        for label, sample_ids in (
            ("support", episode.support_ids),
            ("candidate", episode.candidate_ids),
            ("target", episode.target_ids),
        ):
            missing = set(sample_ids) - known_ids
            if missing:
                raise ValueError(f"Episode {episode.episode_id} has unknown {label} ids: {sorted(missing)[:5]}")
            if not sample_ids:
                raise ValueError(f"Episode {episode.episode_id} requires non-empty {label} ids.")
        role_sets = [
            set(episode.support_group_ids),
            set(episode.candidate_group_ids),
            set(episode.target_group_ids),
        ]
        if role_sets[0] & role_sets[1] or role_sets[0] & role_sets[2] or role_sets[1] & role_sets[2]:
            raise ValueError(f"Episode {episode.episode_id} has source-group leakage between roles.")


def _attach_oracle_fractions(
    round_results: list[RoundResult],
    *,
    episodes: Sequence[EpisodeSpec],
    clips_by_id: dict[str, BenchmarkClip],
    config: OfflineBenchmarkConfig,
) -> None:
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    oracle_by_key: dict[tuple[str, int], float] = {}
    for result in round_results:
        key = (result.episode_id, result.round_index)
        if key in oracle_by_key:
            continue
        episode = episode_by_id[result.episode_id]
        budget = min((int(result.round_index) + 1) * int(config.batch_size), len(episode.candidate_ids))
        oracle_by_key[key] = _exact_oracle_score_for_budget(
            clips_by_id,
            episode=episode,
            config=config,
            budget=budget,
            fallback_results=[
                row
                for row in round_results
                if row.episode_id == result.episode_id and row.round_index == result.round_index
            ],
        )
    for result in round_results:
        oracle = oracle_by_key.get((result.episode_id, result.round_index), 0.0)
        if oracle <= 1.0e-12:
            result.oracle_fraction = 1.0 if abs(float(result.cumulative_balanced_relative_gain)) <= 1.0e-12 else 0.0
        else:
            result.oracle_fraction = max(0.0, min(1.0, float(result.cumulative_balanced_relative_gain) / oracle))


def _exact_oracle_score_for_budget(
    clips_by_id: dict[str, BenchmarkClip],
    *,
    episode: EpisodeSpec,
    config: OfflineBenchmarkConfig,
    budget: int,
    fallback_results: Sequence[RoundResult],
) -> float:
    candidate_ids = tuple(episode.candidate_ids)
    if budget <= 0:
        return 0.0
    if _combination_count(len(candidate_ids), budget) > int(config.oracle_exact_combination_limit):
        return max(float(result.cumulative_balanced_relative_gain) for result in fallback_results)
    best = 0.0
    for selected_ids in combinations(candidate_ids, budget):
        coverage = coverage_report_for_selection(
            clips_by_id,
            support_ids=episode.support_ids,
            target_ids=episode.target_ids,
            candidate_ids=episode.candidate_ids,
            selected_ids=selected_ids,
            representations=config.representations,
        )
        best = max(best, balanced_relative_gain(coverage, primary_representations=config.primary_representations))
    return float(best)


def _combination_count(n_items: int, batch_size: int) -> int:
    if batch_size < 0 or batch_size > n_items:
        return 0
    batch = min(batch_size, n_items - batch_size)
    numerator = 1
    denominator = 1
    for value in range(1, batch + 1):
        numerator *= n_items - (batch - value)
        denominator *= value
    return numerator // denominator


def _summarize_episode_difficulty(
    round_results: Sequence[RoundResult],
    *,
    episodes: Sequence[EpisodeSpec],
    clips_by_id: dict[str, BenchmarkClip],
    config: OfflineBenchmarkConfig,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for episode in episodes:
        baseline = coverage_report_for_selection(
            clips_by_id,
            support_ids=episode.support_ids,
            target_ids=episode.target_ids,
            candidate_ids=episode.candidate_ids,
            selected_ids=(),
            representations=config.representations,
        )
        oracle_rows = sorted(
            [
                result
                for result in round_results
                if result.episode_id == episode.episode_id and result.policy_name == "oracle_greedy_eval_only"
            ],
            key=lambda result: result.round_index,
        )
        oracle_round_gains = [float(result.balanced_relative_gain) for result in oracle_rows]
        oracle_cumulative_gains = [float(result.cumulative_balanced_relative_gain) for result in oracle_rows]
        near_zero_count = sum(abs(value) <= 1.0e-6 for value in oracle_round_gains)
        rows.append(
            {
                "episode_id": episode.episode_id,
                "fold_id": int(episode.fold_id),
                "support_count": int(len(episode.support_ids)),
                "candidate_count": int(len(episode.candidate_ids)),
                "target_count": int(len(episode.target_ids)),
                "support_group_count": int(len(episode.support_group_ids)),
                "candidate_group_count": int(len(episode.candidate_group_ids)),
                "target_group_count": int(len(episode.target_group_ids)),
                "support_target_baseline_distance_by_representation": {
                    representation: float(metrics.get("baseline_distance", 0.0))
                    for representation, metrics in baseline.items()
                },
                "candidate_target_nearest_distance_by_representation": {
                    representation: _mean_nearest_cosine_distance(
                        stack_embeddings(clips_by_id, episode.target_ids, representation=representation),
                        stack_embeddings(clips_by_id, episode.candidate_ids, representation=representation),
                    )
                    for representation in config.representations
                },
                "oracle_greedy_round_gain_by_round": oracle_round_gains,
                "oracle_greedy_cumulative_gain_by_round": oracle_cumulative_gains,
                "max_oracle_greedy_cumulative_gain": max(oracle_cumulative_gains) if oracle_cumulative_gains else 0.0,
                "near_zero_oracle_round_count": int(near_zero_count),
                "near_zero_oracle_round_fraction": float(near_zero_count / len(oracle_round_gains)) if oracle_round_gains else 1.0,
                **_oracle_fraction_exactness_audit(episode, config),
            }
        )
    return tuple(rows)


def _selection_details(
    clips_by_id: dict[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    selected_ids: Sequence[str],
    selected_scores: Sequence[float],
    config: OfflineBenchmarkConfig,
) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    already_selected: list[str] = []
    for rank_index, sample_id in enumerate(selected_ids):
        sample_id = str(sample_id)
        current_support_ids = [*support_ids, *already_selected]
        current_candidate_ids = [str(candidate_id) for candidate_id in candidate_ids if str(candidate_id) not in set(already_selected)]
        clip = clips_by_id[sample_id]
        window_scores = _nearest_novelty_scores(
            clips_by_id,
            support_ids=current_support_ids,
            candidate_ids=current_candidate_ids,
            representation="window",
        )
        left_scores = _nearest_novelty_scores(
            clips_by_id,
            support_ids=current_support_ids,
            candidate_ids=current_candidate_ids,
            representation=str(config.blend_left_representation),
        )
        right_scores = _nearest_novelty_scores(
            clips_by_id,
            support_ids=current_support_ids,
            candidate_ids=current_candidate_ids,
            representation=str(config.blend_right_representation),
        )
        blend_scores = _blend_audit_scores(
            current_candidate_ids,
            left_scores=left_scores,
            right_scores=right_scores,
            alpha=float(config.blend_alpha),
        )
        details.append(
            {
                "rank_index": int(rank_index),
                "sample_id": sample_id,
                "source_group_id": clip.source_group_id,
                "quality_score": float(clip.quality_score),
                "artifact_score": float(clip.artifact_score),
                "stationary_fraction": float(clip.stationary_fraction),
                "max_abs_value": float(clip.max_abs_value),
                "selected_score": float(selected_scores[rank_index]) if rank_index < len(selected_scores) else 0.0,
                "passed_quality_gate": bool(_passes_quality_gate(clip, config)),
                "passed_artifact_gate": bool(_passes_artifact_gate(clip, config)),
                "old_novelty_window_score": _optional_score(window_scores, sample_id),
                "blend_left_representation": str(config.blend_left_representation),
                "blend_left_novelty_score": _optional_score(left_scores, sample_id),
                "blend_right_representation": str(config.blend_right_representation),
                "blend_right_novelty_score": _optional_score(right_scores, sample_id),
                "blend_alpha": float(config.blend_alpha),
                "blend_score": _optional_score(blend_scores, sample_id),
            }
        )
        already_selected.append(sample_id)
    return details


def _nearest_novelty_scores(
    clips_by_id: dict[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    representation: str,
) -> dict[str, float]:
    if not candidate_ids:
        return {}
    first_candidate = clips_by_id[str(candidate_ids[0])]
    if representation not in first_candidate.embeddings:
        return {}
    support = stack_embeddings(clips_by_id, support_ids, representation=representation)
    candidates = stack_embeddings(clips_by_id, candidate_ids, representation=representation)
    support = normalize_rows(support)
    candidates = normalize_rows(candidates)
    if len(support) == 0 or len(candidates) == 0:
        return {str(sample_id): 0.0 for sample_id in candidate_ids}
    nearest = candidates @ support.T
    distances = 1.0 - nearest.max(axis=1)
    return {str(sample_id): float(distances[index]) for index, sample_id in enumerate(candidate_ids)}


def _blend_audit_scores(
    candidate_ids: Sequence[str],
    *,
    left_scores: dict[str, float],
    right_scores: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    if not left_scores or not right_scores:
        return {}
    ids = [str(sample_id) for sample_id in candidate_ids if str(sample_id) in left_scores and str(sample_id) in right_scores]
    if not ids:
        return {}
    left = _minmax_scores([left_scores[sample_id] for sample_id in ids])
    right = _minmax_scores([right_scores[sample_id] for sample_id in ids])
    weight = float(alpha)
    return {
        sample_id: float(weight * left[index] + (1.0 - weight) * right[index])
        for index, sample_id in enumerate(ids)
    }


def _minmax_scores(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return array
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum <= 1.0e-12:
        return np.zeros_like(array, dtype=float)
    return (array - minimum) / (maximum - minimum)


def _optional_score(scores: dict[str, float], sample_id: str) -> float | None:
    if sample_id not in scores:
        return None
    return float(scores[sample_id])


def _passes_quality_gate(clip: BenchmarkClip, config: OfflineBenchmarkConfig) -> bool:
    return (
        float(clip.quality_score) >= float(config.quality_threshold)
        and float(clip.stationary_fraction) <= float(config.max_stationary_fraction)
        and float(clip.max_abs_value) <= float(config.max_abs_value)
    )


def _passes_artifact_gate(clip: BenchmarkClip, config: OfflineBenchmarkConfig) -> bool:
    return config.max_artifact_score is None or float(clip.artifact_score) <= float(config.max_artifact_score)


def _oracle_fraction_exactness_audit(episode: EpisodeSpec, config: OfflineBenchmarkConfig) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    candidate_count = len(episode.candidate_ids)
    for round_index in range(int(config.rounds)):
        budget = min((round_index + 1) * int(config.batch_size), candidate_count)
        combination_count = _combination_count(candidate_count, budget)
        exact = combination_count <= int(config.oracle_exact_combination_limit)
        rows.append(
            {
                "round_index": int(round_index),
                "budget": int(budget),
                "candidate_count": int(candidate_count),
                "combination_count": int(combination_count),
                "exact": bool(exact),
                "oracle_exact_combination_limit": int(config.oracle_exact_combination_limit),
            }
        )
    exact_by_round = [bool(row["exact"]) for row in rows]
    return {
        "oracle_fraction_budget_audit": rows,
        "oracle_fraction_exact_by_round": exact_by_round,
        "oracle_fraction_exact_all_rounds": bool(all(exact_by_round)) if exact_by_round else False,
    }


def _mean_nearest_cosine_distance(query_embeddings, support_embeddings) -> float:
    query = normalize_rows(query_embeddings)
    support = normalize_rows(support_embeddings)
    if len(query) == 0 or len(support) == 0:
        return 0.0
    similarities = query @ support.T
    nearest = similarities.max(axis=1)
    return float((1.0 - nearest).mean())


def _emit_progress(
    progress_callback: Callable[[dict[str, object]], None] | None,
    event: str,
    **payload: object,
) -> None:
    if progress_callback is None:
        return
    progress_callback({"event": event, **payload})
