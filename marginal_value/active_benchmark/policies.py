from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from marginal_value.active_benchmark.metrics import balanced_relative_gain, coverage_report_for_selection
from marginal_value.active_benchmark.representations import stack_embeddings
from marginal_value.active_benchmark.schema import BenchmarkClip, OfflineBenchmarkConfig
from marginal_value.indexing.knn_features import normalize_rows


SUPPORTED_POLICIES = (
    "random_valid",
    "quality_only",
    "quality_stratified_random",
    "old_novelty_window",
    "old_novelty_window_sourcecap2",
    "old_novelty_ts2vec",
    "kcenter_quality_gated_window",
    "kcenter_quality_gated_ts2vec",
    "blend_kcenter_ts2vec_window",
    "artifact_gate_blend_kcenter_ts2vec_window",
    "submitted_full_replay",
    "submitted_minus_ts2vec",
    "submitted_minus_window",
    "submitted_no_kcenter",
    "window_novelty_same_gates_no_kcenter",
    "ts2vec_novelty_same_gates_no_kcenter",
    "oracle_greedy_eval_only",
)
RANDOM_REPLAY_PREFIX = "random_valid_replay_"


def select_batch(
    policy_name: str,
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    target_ids: Sequence[str],
    config: OfflineBenchmarkConfig,
    episode_index: int,
    round_index: int,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    random_replay_index = _random_replay_index(policy_name)
    if policy_name not in SUPPORTED_POLICIES and random_replay_index is None:
        raise ValueError(f"Unsupported offline active benchmark policy: {policy_name}")
    batch_size = min(max(0, int(config.batch_size)), len(candidate_ids))
    if batch_size == 0:
        return (), ()
    if policy_name == "random_valid" or random_replay_index is not None:
        return _random_valid(
            clips_by_id,
            candidate_ids,
            config=config,
            episode_index=episode_index,
            round_index=round_index,
            batch_size=batch_size,
            replay_index=random_replay_index,
        )
    if policy_name == "quality_only":
        order = _quality_order(clips_by_id, candidate_ids)
        return _selected_with_scores(order[:batch_size], clips_by_id, score_fn=lambda clip: float(clip.quality_score))
    if policy_name == "quality_stratified_random":
        return _quality_stratified_random(
            clips_by_id,
            candidate_ids,
            config=config,
            episode_index=episode_index,
            round_index=round_index,
            batch_size=batch_size,
        )
    if policy_name == "old_novelty_window":
        scores = _old_novelty_scores(clips_by_id, support_ids=support_ids, candidate_ids=candidate_ids)
        order = _score_order(candidate_ids, scores, clips_by_id)
        return _selected_with_scores(order[:batch_size], clips_by_id, score_fn=lambda clip: scores[clip.sample_id])
    if policy_name == "old_novelty_window_sourcecap2":
        scores = _old_novelty_scores(clips_by_id, support_ids=support_ids, candidate_ids=candidate_ids)
        order = _score_order(candidate_ids, scores, clips_by_id)
        order = _source_capped_order(order, clips_by_id, source_cap=2)
        return _selected_with_scores(order[:batch_size], clips_by_id, score_fn=lambda clip: scores[clip.sample_id])
    if policy_name == "old_novelty_ts2vec":
        scores = _novelty_scores_by_representation(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            representation=str(config.blend_left_representation),
        )
        order = _score_order(candidate_ids, scores, clips_by_id)
        return _selected_with_scores(order[:batch_size], clips_by_id, score_fn=lambda clip: scores[clip.sample_id])
    if policy_name == "kcenter_quality_gated_window":
        return _kcenter_quality_gated(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            config=config,
            batch_size=batch_size,
            representation="window",
        )
    if policy_name == "kcenter_quality_gated_ts2vec":
        return _kcenter_quality_gated(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            config=config,
            batch_size=batch_size,
            representation=str(config.blend_left_representation),
        )
    if policy_name in {
        "blend_kcenter_ts2vec_window",
        "artifact_gate_blend_kcenter_ts2vec_window",
        "submitted_full_replay",
    }:
        order, scores = _blended_kcenter_quality_gated_order(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            config=config,
        )
        if policy_name in {"artifact_gate_blend_kcenter_ts2vec_window", "submitted_full_replay"}:
            order = _artifact_gate_order(order, clips_by_id, config=config)
        return _selected_with_scores(order[:batch_size], clips_by_id, score_fn=lambda clip: scores[clip.sample_id])
    if policy_name == "submitted_minus_ts2vec":
        selected, selected_scores = _kcenter_quality_gated(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            config=config,
            batch_size=batch_size,
            representation=str(config.blend_right_representation),
        )
        return _artifact_gated_selected_with_scores(selected, selected_scores, clips_by_id, config=config)
    if policy_name == "submitted_minus_window":
        selected, selected_scores = _kcenter_quality_gated(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            config=config,
            batch_size=batch_size,
            representation=str(config.blend_left_representation),
        )
        return _artifact_gated_selected_with_scores(selected, selected_scores, clips_by_id, config=config)
    if policy_name == "submitted_no_kcenter":
        order, scores = _blended_novelty_quality_gated_order(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            config=config,
        )
        order = _artifact_gate_order(order, clips_by_id, config=config)
        return _selected_with_scores(order[:batch_size], clips_by_id, score_fn=lambda clip: scores[clip.sample_id])
    if policy_name == "window_novelty_same_gates_no_kcenter":
        order, scores = _novelty_quality_gated_order(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            config=config,
            representation=str(config.blend_right_representation),
        )
        return _selected_with_scores(order[:batch_size], clips_by_id, score_fn=lambda clip: scores[clip.sample_id])
    if policy_name == "ts2vec_novelty_same_gates_no_kcenter":
        order, scores = _novelty_quality_gated_order(
            clips_by_id,
            support_ids=support_ids,
            candidate_ids=candidate_ids,
            config=config,
            representation=str(config.blend_left_representation),
        )
        return _selected_with_scores(order[:batch_size], clips_by_id, score_fn=lambda clip: scores[clip.sample_id])
    return _oracle_greedy(
        clips_by_id,
        support_ids=support_ids,
        candidate_ids=candidate_ids,
        target_ids=target_ids,
        config=config,
        batch_size=batch_size,
    )


def _random_valid(
    clips_by_id: Mapping[str, BenchmarkClip],
    candidate_ids: Sequence[str],
    *,
    config: OfflineBenchmarkConfig,
    episode_index: int,
    round_index: int,
    batch_size: int,
    replay_index: int | None = None,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    valid = [sample_id for sample_id in candidate_ids if _passes_gates(clips_by_id[str(sample_id)], config)]
    fallback = [sample_id for sample_id in candidate_ids if sample_id not in set(valid)]
    replay_offset = 0 if replay_index is None else (int(replay_index) + 1) * 104729
    rng = np.random.default_rng(int(config.random_seed) + episode_index * 997 + round_index * 31 + replay_offset)
    valid_order = list(rng.permutation(valid)) if valid else []
    fallback_order = list(rng.permutation(fallback)) if fallback else []
    selected = tuple(str(sample_id) for sample_id in [*valid_order, *fallback_order][:batch_size])
    return selected, tuple(float(clips_by_id[sample_id].quality_score) for sample_id in selected)


def _quality_stratified_random(
    clips_by_id: Mapping[str, BenchmarkClip],
    candidate_ids: Sequence[str],
    *,
    config: OfflineBenchmarkConfig,
    episode_index: int,
    round_index: int,
    batch_size: int,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    eligible = [str(sample_id) for sample_id in candidate_ids if _passes_gates(clips_by_id[str(sample_id)], config)]
    eligible_set = set(eligible)
    fallback = [str(sample_id) for sample_id in candidate_ids if str(sample_id) not in eligible_set]
    quality_order = _quality_order(clips_by_id, eligible)
    stratum_size = min(len(quality_order), max(batch_size * 2, batch_size))
    stratum = quality_order[:stratum_size]
    remainder = quality_order[stratum_size:]

    rng = np.random.default_rng(int(config.random_seed) + episode_index * 997 + round_index * 31 + 65537)
    stratum_order = list(rng.permutation(stratum)) if stratum else []
    remainder_order = list(rng.permutation(remainder)) if remainder else []
    fallback_order = _quality_order(clips_by_id, fallback)
    selected = tuple(str(sample_id) for sample_id in [*stratum_order, *remainder_order, *fallback_order][:batch_size])
    return selected, tuple(float(clips_by_id[sample_id].quality_score) for sample_id in selected)


def _random_replay_index(policy_name: str) -> int | None:
    if not policy_name.startswith(RANDOM_REPLAY_PREFIX):
        return None
    suffix = policy_name[len(RANDOM_REPLAY_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _quality_order(clips_by_id: Mapping[str, BenchmarkClip], candidate_ids: Sequence[str]) -> list[str]:
    return sorted(
        [str(sample_id) for sample_id in candidate_ids],
        key=lambda sample_id: (-float(clips_by_id[sample_id].quality_score), sample_id),
    )


def _old_novelty_scores(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    return _novelty_scores_by_representation(
        clips_by_id,
        support_ids=support_ids,
        candidate_ids=candidate_ids,
        representation="window",
    )


def _novelty_scores_by_representation(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    representation: str,
) -> dict[str, float]:
    support = stack_embeddings(clips_by_id, support_ids, representation=representation)
    candidates = stack_embeddings(clips_by_id, candidate_ids, representation=representation)
    distances = _nearest_cosine_distances(support, candidates)
    return {str(sample_id): float(distances[index]) for index, sample_id in enumerate(candidate_ids)}


def _kcenter_quality_gated(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    config: OfflineBenchmarkConfig,
    batch_size: int,
    representation: str,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    eligible = [str(sample_id) for sample_id in candidate_ids if _passes_gates(clips_by_id[str(sample_id)], config)]
    fallback = [str(sample_id) for sample_id in candidate_ids if sample_id not in set(eligible)]
    selected: list[str] = []
    remaining = list(eligible)
    scores: dict[str, float] = {}
    while remaining and len(selected) < batch_size:
        support_plus_selected = [*support_ids, *selected]
        support = stack_embeddings(clips_by_id, support_plus_selected, representation=representation)
        candidates = stack_embeddings(clips_by_id, remaining, representation=representation)
        distances = _nearest_cosine_distances(support, candidates)
        by_id = {sample_id: float(distances[index]) for index, sample_id in enumerate(remaining)}
        best = max(
            remaining,
            key=lambda sample_id: (
                by_id[sample_id],
                float(clips_by_id[sample_id].quality_score),
                sample_id,
            ),
        )
        scores[best] = by_id[best]
        selected.append(best)
        remaining.remove(best)
    for sample_id in fallback:
        if len(selected) >= batch_size:
            break
        scores[sample_id] = float(clips_by_id[sample_id].quality_score)
        selected.append(sample_id)
    return tuple(selected), tuple(scores[sample_id] for sample_id in selected)


def _novelty_quality_gated_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    config: OfflineBenchmarkConfig,
    representation: str,
) -> tuple[list[str], dict[str, float]]:
    eligible = [str(sample_id) for sample_id in candidate_ids if _passes_gates(clips_by_id[str(sample_id)], config)]
    fallback = [str(sample_id) for sample_id in candidate_ids if sample_id not in set(eligible)]
    scores = _novelty_scores_by_representation(
        clips_by_id,
        support_ids=support_ids,
        candidate_ids=eligible,
        representation=representation,
    )
    order = _score_order(eligible, scores, clips_by_id)
    fallback_order = _quality_order(clips_by_id, fallback)
    for sample_id in fallback_order:
        scores[sample_id] = float(clips_by_id[sample_id].quality_score)
    return [*order, *fallback_order], scores


def _blended_novelty_quality_gated_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    config: OfflineBenchmarkConfig,
) -> tuple[list[str], dict[str, float]]:
    eligible = [str(sample_id) for sample_id in candidate_ids if _passes_gates(clips_by_id[str(sample_id)], config)]
    fallback = [str(sample_id) for sample_id in candidate_ids if sample_id not in set(eligible)]
    scores = _blended_distance_scores(
        clips_by_id,
        support_ids=support_ids,
        candidate_ids=eligible,
        left_representation=str(config.blend_left_representation),
        right_representation=str(config.blend_right_representation),
        alpha=float(config.blend_alpha),
    )
    order = _score_order(eligible, scores, clips_by_id)
    fallback_order = _quality_order(clips_by_id, fallback)
    for sample_id in fallback_order:
        scores[sample_id] = float(clips_by_id[sample_id].quality_score)
    return [*order, *fallback_order], scores


def _blended_kcenter_quality_gated_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    config: OfflineBenchmarkConfig,
) -> tuple[list[str], dict[str, float]]:
    eligible = [str(sample_id) for sample_id in candidate_ids if _passes_gates(clips_by_id[str(sample_id)], config)]
    fallback = [str(sample_id) for sample_id in candidate_ids if sample_id not in set(eligible)]
    selected: list[str] = []
    remaining = list(eligible)
    scores: dict[str, float] = {}
    while remaining:
        score_by_id = _blended_distance_scores(
            clips_by_id,
            support_ids=[*support_ids, *selected],
            candidate_ids=remaining,
            left_representation=str(config.blend_left_representation),
            right_representation=str(config.blend_right_representation),
            alpha=float(config.blend_alpha),
        )
        best = max(
            remaining,
            key=lambda sample_id: (
                score_by_id[sample_id],
                float(clips_by_id[sample_id].quality_score),
                sample_id,
            ),
        )
        scores[best] = float(score_by_id[best])
        selected.append(best)
        remaining.remove(best)
    fallback_order = _quality_order(clips_by_id, fallback)
    for sample_id in fallback_order:
        scores[sample_id] = float(clips_by_id[sample_id].quality_score)
    return [*selected, *fallback_order], scores


def _blended_distance_scores(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    left_representation: str,
    right_representation: str,
    alpha: float,
) -> dict[str, float]:
    left_support = stack_embeddings(clips_by_id, support_ids, representation=left_representation)
    left_candidates = stack_embeddings(clips_by_id, candidate_ids, representation=left_representation)
    right_support = stack_embeddings(clips_by_id, support_ids, representation=right_representation)
    right_candidates = stack_embeddings(clips_by_id, candidate_ids, representation=right_representation)
    left_distances = _nearest_cosine_distances(left_support, left_candidates)
    right_distances = _nearest_cosine_distances(right_support, right_candidates)
    weight = float(alpha)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("blend_alpha must be in [0, 1].")
    blended = weight * _minmax(left_distances) + (1.0 - weight) * _minmax(right_distances)
    return {str(sample_id): float(blended[index]) for index, sample_id in enumerate(candidate_ids)}


def _artifact_gate_order(
    order: Sequence[str],
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    config: OfflineBenchmarkConfig,
) -> list[str]:
    threshold = config.max_artifact_score
    if threshold is None:
        return [str(sample_id) for sample_id in order]
    passed: list[str] = []
    failed: list[str] = []
    for sample_id in order:
        target = failed if float(clips_by_id[str(sample_id)].artifact_score) > float(threshold) else passed
        target.append(str(sample_id))
    return [*passed, *failed]


def _artifact_gated_selected_with_scores(
    selected: Sequence[str],
    selected_scores: Sequence[float],
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    config: OfflineBenchmarkConfig,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    scores_by_id = {str(sample_id): float(selected_scores[index]) for index, sample_id in enumerate(selected)}
    gated = _artifact_gate_order(selected, clips_by_id, config=config)
    return tuple(gated), tuple(scores_by_id[sample_id] for sample_id in gated)


def _source_capped_order(
    order: Sequence[str],
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    source_cap: int,
) -> list[str]:
    if int(source_cap) <= 0:
        return [str(sample_id) for sample_id in order]
    counts: dict[str, int] = {}
    under_cap: list[str] = []
    over_cap: list[str] = []
    for sample_id in order:
        sample_id = str(sample_id)
        source_group = str(clips_by_id[sample_id].source_group_id)
        count = counts.get(source_group, 0)
        if count < int(source_cap):
            under_cap.append(sample_id)
        else:
            over_cap.append(sample_id)
        counts[source_group] = count + 1
    return [*under_cap, *over_cap]


def _oracle_greedy(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    target_ids: Sequence[str],
    config: OfflineBenchmarkConfig,
    batch_size: int,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    candidate_ids = _screen_oracle_candidates(
        clips_by_id,
        support_ids=support_ids,
        candidate_ids=candidate_ids,
        target_ids=target_ids,
        config=config,
    )
    exact = _oracle_exact_when_small(
        clips_by_id,
        support_ids=support_ids,
        candidate_ids=candidate_ids,
        target_ids=target_ids,
        config=config,
        batch_size=batch_size,
    )
    if exact is not None:
        return exact

    selected: list[str] = []
    remaining = [str(sample_id) for sample_id in candidate_ids]
    scores: dict[str, float] = {}
    while remaining and len(selected) < batch_size:
        best_score_by_id: dict[str, float] = {}
        for sample_id in remaining:
            trial_selected = [*selected, sample_id]
            coverage = coverage_report_for_selection(
                clips_by_id,
                support_ids=support_ids,
                target_ids=target_ids,
                candidate_ids=candidate_ids,
                selected_ids=trial_selected,
                representations=config.representations,
            )
            best_score_by_id[sample_id] = balanced_relative_gain(
                coverage,
                primary_representations=config.primary_representations,
            )
        best = max(best_score_by_id, key=lambda sample_id: (best_score_by_id[sample_id], sample_id))
        scores[best] = best_score_by_id[best]
        selected.append(best)
        remaining.remove(best)
    return tuple(selected), tuple(scores[sample_id] for sample_id in selected)


def _oracle_exact_when_small(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    target_ids: Sequence[str],
    config: OfflineBenchmarkConfig,
    batch_size: int,
) -> tuple[tuple[str, ...], tuple[float, ...]] | None:
    candidates = [str(sample_id) for sample_id in candidate_ids]
    if batch_size <= 0 or len(candidates) < batch_size:
        return None
    combination_count = _combination_count(len(candidates), batch_size)
    if combination_count > int(config.oracle_exact_combination_limit):
        return None
    best_selected: tuple[str, ...] | None = None
    best_score = float("-inf")
    for selected in combinations(candidates, batch_size):
        coverage = coverage_report_for_selection(
            clips_by_id,
            support_ids=support_ids,
            target_ids=target_ids,
            candidate_ids=candidate_ids,
            selected_ids=selected,
            representations=config.representations,
        )
        score = balanced_relative_gain(coverage, primary_representations=config.primary_representations)
        if score > best_score or (abs(score - best_score) <= 1.0e-12 and tuple(selected) > (best_selected or ())):
            best_score = score
            best_selected = tuple(selected)
    if best_selected is None:
        return None
    return best_selected, tuple(float(best_score) for _sample_id in best_selected)


def _screen_oracle_candidates(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    target_ids: Sequence[str],
    config: OfflineBenchmarkConfig,
) -> tuple[str, ...]:
    cap = config.oracle_candidate_cap
    candidates = tuple(str(sample_id) for sample_id in candidate_ids)
    if cap is None or int(cap) <= 0 or len(candidates) <= int(cap):
        return candidates
    rows: list[tuple[float, float, str]] = []
    for sample_id in candidates:
        coverage = coverage_report_for_selection(
            clips_by_id,
            support_ids=support_ids,
            target_ids=target_ids,
            candidate_ids=candidates,
            selected_ids=(sample_id,),
            representations=config.representations,
        )
        rows.append(
            (
                balanced_relative_gain(coverage, primary_representations=config.primary_representations),
                float(clips_by_id[sample_id].quality_score),
                sample_id,
            )
        )
    return tuple(sample_id for _gain, _quality, sample_id in sorted(rows, key=lambda row: (-row[0], -row[1], row[2]))[: int(cap)])


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


def _score_order(candidate_ids: Sequence[str], scores: Mapping[str, float], clips_by_id: Mapping[str, BenchmarkClip]) -> list[str]:
    return sorted(
        [str(sample_id) for sample_id in candidate_ids],
        key=lambda sample_id: (
            -float(scores[sample_id]),
            -float(clips_by_id[sample_id].quality_score),
            sample_id,
        ),
    )


def _selected_with_scores(
    selected: Sequence[str],
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    score_fn,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    selected_ids = tuple(str(sample_id) for sample_id in selected)
    return selected_ids, tuple(float(score_fn(clips_by_id[sample_id])) for sample_id in selected_ids)


def _passes_gates(clip: BenchmarkClip, config: OfflineBenchmarkConfig) -> bool:
    return (
        float(clip.quality_score) >= float(config.quality_threshold)
        and float(clip.stationary_fraction) <= float(config.max_stationary_fraction)
        and float(clip.max_abs_value) <= float(config.max_abs_value)
    )


def _nearest_cosine_distances(support_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    support = normalize_rows(np.asarray(support_embeddings, dtype=float))
    candidates = normalize_rows(np.asarray(candidate_embeddings, dtype=float))
    if len(support) == 0 or len(candidates) == 0:
        return np.zeros((len(candidates),), dtype=float)
    similarities = candidates @ support.T
    nearest = np.max(similarities, axis=1)
    return np.maximum(1.0 - nearest, 0.0)


def _minmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    lo = float(np.min(array))
    hi = float(np.max(array))
    if hi - lo < 1.0e-12:
        return np.zeros_like(array, dtype=float)
    return (array - lo) / (hi - lo)
