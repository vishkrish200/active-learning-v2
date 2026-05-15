from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from marginal_value.active_benchmark.representations import stack_embeddings
from marginal_value.active_benchmark.schema import BenchmarkClip
from marginal_value.eval.marginal_coverage_eval import coverage_gain_for_selection


def coverage_report_for_selection(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    target_ids: Sequence[str],
    candidate_ids: Sequence[str],
    selected_ids: Sequence[str],
    representations: Sequence[str],
) -> dict[str, dict[str, float]]:
    selected_set = {str(sample_id) for sample_id in selected_ids}
    selected_indices = [
        index
        for index, sample_id in enumerate(candidate_ids)
        if str(sample_id) in selected_set
    ]
    report: dict[str, dict[str, float]] = {}
    for representation in representations:
        support = stack_embeddings(clips_by_id, support_ids, representation=representation)
        target = stack_embeddings(clips_by_id, target_ids, representation=representation)
        candidates = stack_embeddings(clips_by_id, candidate_ids, representation=representation)
        report[str(representation)] = coverage_gain_for_selection(
            support,
            target,
            candidates,
            selected_indices=selected_indices,
        )
    return report


def balanced_relative_gain(
    coverage_by_representation: Mapping[str, Mapping[str, float]],
    *,
    primary_representations: Sequence[str],
) -> float:
    values = [
        float(coverage_by_representation.get(rep, {}).get("relative_coverage_gain", 0.0))
        for rep in primary_representations
    ]
    return float(np.mean(values)) if values else 0.0


def selection_summary(
    clips_by_id: Mapping[str, BenchmarkClip],
    selected_ids: Sequence[str],
) -> dict[str, float | int]:
    clips = [clips_by_id[str(sample_id)] for sample_id in selected_ids]
    if not clips:
        return {
            "n_selected": 0,
            "mean_quality": 0.0,
            "min_quality": 0.0,
            "low_quality_rate": 0.0,
            "artifact_rate": 0.0,
            "duplicate_rate": 0.0,
            "unique_source_groups": 0,
            "largest_source_group_fraction": 0.0,
            "max_stationary_fraction": 0.0,
            "max_abs_value": 0.0,
        }
    groups = [clip.source_group_id for clip in clips]
    qualities = [float(clip.quality_score) for clip in clips]
    group_counts = Counter(groups)
    duplicate_count = len(clips) - len(set(clip.sample_id for clip in clips))
    return {
        "n_selected": int(len(clips)),
        "mean_quality": float(np.mean(qualities)),
        "min_quality": float(np.min(qualities)),
        "low_quality_rate": float(np.mean([quality < 0.85 for quality in qualities])),
        "artifact_rate": float(np.mean([float(clip.artifact_score) > 0.0 for clip in clips])),
        "duplicate_rate": float(duplicate_count / len(clips)),
        "unique_source_groups": int(len(group_counts)),
        "largest_source_group_fraction": float(max(group_counts.values()) / len(clips)),
        "max_stationary_fraction": float(max(float(clip.stationary_fraction) for clip in clips)),
        "max_abs_value": float(max(float(clip.max_abs_value) for clip in clips)),
    }


def summarize_policy_rounds(rounds) -> dict[str, dict[str, float]]:
    by_policy: dict[str, list[float]] = {}
    wins_by_policy: dict[str, int] = {}
    for result in rounds:
        by_policy.setdefault(result.policy_name, []).append(float(result.balanced_relative_gain))
        wins_by_policy.setdefault(result.policy_name, 0)

    grouped_by_episode_round: dict[tuple[str, int], list] = {}
    for result in rounds:
        grouped_by_episode_round.setdefault((result.episode_id, result.round_index), []).append(result)
    for group in grouped_by_episode_round.values():
        best = max(float(result.balanced_relative_gain) for result in group)
        for result in group:
            if abs(float(result.balanced_relative_gain) - best) <= 1.0e-12:
                wins_by_policy[result.policy_name] = wins_by_policy.get(result.policy_name, 0) + 1

    return {
        policy: {
            "mean_balanced_relative_gain": float(np.mean(values)) if values else 0.0,
            "round_count": float(len(values)),
            "episode_round_wins": float(wins_by_policy.get(policy, 0)),
        }
        for policy, values in sorted(by_policy.items())
    }
