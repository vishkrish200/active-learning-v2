from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from marginal_value.active_benchmark.representations import clip_lookup, stack_embeddings
from marginal_value.active_benchmark.schema import BenchmarkClip, EpisodeSpec
from marginal_value.active_benchmark.splits import infer_source_family_assignments
from marginal_value.indexing.knn_features import normalize_rows


SUPPORTED_COVERAGE_POLICIES = (
    "random_valid_v1",
    "quality_stratified_random_v1",
    "quality_only_v1",
    "window_support_novelty_v1",
    "window_kcenter_v1",
    "ts2vec_support_novelty_v1",
    "ts2vec_kcenter_v1",
    "submitted_full_replay_v1",
    "oracle_greedy_eval_view_v1",
    "oracle_greedy_target_family_v1",
)
HYGIENE_EVAL_VIEW = "__hygiene__"


@dataclass(frozen=True)
class CoverageBenchmarkConfig:
    budgets: Sequence[int]
    policies: Sequence[str]
    eval_views: Sequence[str]
    primary_eval_views: Sequence[str]
    random_seed: int = 17
    quality_threshold: float = 0.85
    max_stationary_fraction: float = 0.90
    max_abs_value: float = 60.0
    max_artifact_score: float | None = 0.05
    ts2vec_view: str = "ts2vec"
    window_view: str = "window"
    blend_alpha: float = 0.5
    eval_view_families: Mapping[str, str] = field(default_factory=dict)
    distance_metric: str = "euclidean"
    source_family_count: int = 4
    source_family_label_view: str = "window"
    eps: float = 1.0e-12


@dataclass(frozen=True)
class CoverageRoundRow:
    episode_id: str
    fold_id: int
    policy_id: str
    budget_k: int
    selected_count: int
    candidate_count: int
    eligible_candidate_count: int
    status: str
    primary_coverage_gain_rel: float


@dataclass(frozen=True)
class CoverageSelectedRow:
    episode_id: str
    fold_id: int
    policy_id: str
    budget_k: int
    rank_index: int
    sample_id: str
    source_group_id: str
    score: float
    quality_score: float
    artifact_score: float
    valid: bool
    passed_artifact_gate: bool


@dataclass(frozen=True)
class CoverageMetricRow:
    episode_id: str
    fold_id: int
    policy_id: str
    budget_k: int
    eval_view: str
    metric_name: str
    metric_value: float
    higher_is_better: bool
    primary_eval: bool
    selector_feature_overlap: bool
    uses_target_for_selection: bool


@dataclass(frozen=True)
class CoverageRunResult:
    episodes: tuple[EpisodeSpec, ...]
    policies: tuple[str, ...]
    budgets: tuple[int, ...]
    rounds: tuple[CoverageRoundRow, ...]
    selected_rows: tuple[CoverageSelectedRow, ...]
    metric_rows: tuple[CoverageMetricRow, ...]
    policy_summary: dict[str, dict[str, float]]
    config: CoverageBenchmarkConfig


@dataclass(frozen=True)
class _PolicySpec:
    policy_id: str
    selector_view: str | None
    selector_feature_families: tuple[str, ...]
    uses_target_for_selection: bool = False


def run_coverage_benchmark(
    clips: Sequence[BenchmarkClip],
    episodes: Sequence[EpisodeSpec],
    config: CoverageBenchmarkConfig,
    *,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> CoverageRunResult:
    clips_by_id = clip_lookup(clips)
    budgets = _validated_budgets(config)
    policies = _validated_policies(config)
    _validate_config(config, policies=policies)
    _validate_episodes(episodes, clips_by_id)

    round_rows: list[CoverageRoundRow] = []
    selected_rows: list[CoverageSelectedRow] = []
    metric_rows: list[CoverageMetricRow] = []

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
        for policy_index, policy_id in enumerate(policies):
            policy_spec = _policy_spec(policy_id, config)
            order, scores = _rank_candidates(
                policy_spec,
                clips_by_id,
                episode=episode,
                config=config,
                episode_index=episode_index,
                policy_index=policy_index,
            )
            eligible_count = len(_eligible_candidates(clips_by_id, episode.candidate_ids, config))
            for budget_k in budgets:
                selected_ids = tuple(order[: int(budget_k)])
                metric_rows.extend(
                    _coverage_metric_rows(
                        clips_by_id,
                        episode=episode,
                        selected_ids=selected_ids,
                        policy_spec=policy_spec,
                        config=config,
                        budget_k=int(budget_k),
                    )
                )
                metric_rows.extend(
                    _hygiene_metric_rows(
                        clips_by_id,
                        episode=episode,
                        selected_ids=selected_ids,
                        policy_spec=policy_spec,
                        config=config,
                        budget_k=int(budget_k),
                    )
                )
                selected_rows.extend(
                    _selected_rows(
                        clips_by_id,
                        episode=episode,
                        selected_ids=selected_ids,
                        scores=scores,
                        policy_spec=policy_spec,
                        config=config,
                        budget_k=int(budget_k),
                    )
                )
                primary_gain = _primary_gain(metric_rows, episode.episode_id, policy_id, int(budget_k))
                round_rows.append(
                    CoverageRoundRow(
                        episode_id=episode.episode_id,
                        fold_id=int(episode.fold_id),
                        policy_id=policy_id,
                        budget_k=int(budget_k),
                        selected_count=len(selected_ids),
                        candidate_count=len(episode.candidate_ids),
                        eligible_candidate_count=eligible_count,
                        status="ok" if len(selected_ids) >= int(budget_k) else "underfilled",
                        primary_coverage_gain_rel=float(primary_gain),
                    )
                )
                _emit_progress(
                    progress_callback,
                    "budget_done",
                    episode_id=episode.episode_id,
                    fold_id=int(episode.fold_id),
                    policy_id=policy_id,
                    budget_k=int(budget_k),
                    selected_count=len(selected_ids),
                    primary_coverage_gain_rel=float(primary_gain),
                )

    result = CoverageRunResult(
        episodes=tuple(episodes),
        policies=policies,
        budgets=budgets,
        rounds=tuple(round_rows),
        selected_rows=tuple(selected_rows),
        metric_rows=tuple(metric_rows),
        policy_summary=_policy_summary(metric_rows, round_rows),
        config=config,
    )
    _emit_progress(
        progress_callback,
        "coverage_benchmark_done",
        n_episodes=len(result.episodes),
        n_round_rows=len(result.rounds),
        n_metric_rows=len(result.metric_rows),
        policies=list(result.policies),
    )
    return result


def _validated_budgets(config: CoverageBenchmarkConfig) -> tuple[int, ...]:
    budgets = tuple(sorted({int(budget) for budget in config.budgets}))
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("Coverage benchmark requires positive budgets.")
    return budgets


def _validated_policies(config: CoverageBenchmarkConfig) -> tuple[str, ...]:
    policies = tuple(str(policy) for policy in config.policies)
    if not policies:
        raise ValueError("Coverage benchmark requires at least one policy.")
    unknown = sorted(set(policies) - set(SUPPORTED_COVERAGE_POLICIES))
    if unknown:
        raise ValueError(f"Unsupported coverage benchmark policies: {unknown}")
    return policies


def _validate_config(config: CoverageBenchmarkConfig, *, policies: Sequence[str]) -> None:
    if not config.eval_views:
        raise ValueError("Coverage benchmark requires at least one eval view.")
    if not config.primary_eval_views:
        raise ValueError("Coverage benchmark requires at least one primary eval view.")
    missing_primary = set(config.primary_eval_views) - set(config.eval_views)
    if missing_primary:
        raise ValueError(f"Primary eval views are not loaded as eval views: {sorted(missing_primary)}")
    if config.distance_metric not in {"euclidean", "cosine"}:
        raise ValueError("Coverage benchmark distance_metric must be 'euclidean' or 'cosine'.")
    if not 0.0 <= float(config.blend_alpha) <= 1.0:
        raise ValueError("Coverage benchmark blend_alpha must be in [0, 1].")
    if config.max_artifact_score is not None and float(config.max_artifact_score) < 0.0:
        raise ValueError("Coverage benchmark max_artifact_score must be non-negative when provided.")
    if any(policy in {"window_support_novelty_v1", "window_kcenter_v1", "submitted_full_replay_v1"} for policy in policies):
        if not str(config.window_view):
            raise ValueError("Window coverage policies require window_view.")
    if any(policy in {"ts2vec_support_novelty_v1", "ts2vec_kcenter_v1", "submitted_full_replay_v1"} for policy in policies):
        if not str(config.ts2vec_view):
            raise ValueError("TS2Vec coverage policies require ts2vec_view.")
    if "submitted_full_replay_v1" in policies and not str(config.window_view):
        raise ValueError("submitted_full_replay_v1 requires window_view.")


def _validate_episodes(episodes: Sequence[EpisodeSpec], clips_by_id: Mapping[str, BenchmarkClip]) -> None:
    if not episodes:
        raise ValueError("Coverage benchmark requires at least one episode.")
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
            if len(set(sample_ids)) != len(tuple(sample_ids)):
                raise ValueError(f"Episode {episode.episode_id} has duplicate {label} ids.")
        role_id_sets = (set(episode.support_ids), set(episode.candidate_ids), set(episode.target_ids))
        if role_id_sets[0] & role_id_sets[1] or role_id_sets[0] & role_id_sets[2] or role_id_sets[1] & role_id_sets[2]:
            raise ValueError(f"Episode {episode.episode_id} has sample-id leakage between roles.")
        role_group_sets = (
            set(episode.support_group_ids),
            set(episode.candidate_group_ids),
            set(episode.target_group_ids),
        )
        if (
            role_group_sets[0] & role_group_sets[1]
            or role_group_sets[0] & role_group_sets[2]
            or role_group_sets[1] & role_group_sets[2]
        ):
            raise ValueError(f"Episode {episode.episode_id} has source-group leakage between roles.")


def _policy_spec(policy_id: str, config: CoverageBenchmarkConfig) -> _PolicySpec:
    if policy_id == "random_valid_v1":
        return _PolicySpec(policy_id=policy_id, selector_view=None, selector_feature_families=("random",))
    if policy_id == "quality_stratified_random_v1":
        return _PolicySpec(policy_id=policy_id, selector_view=None, selector_feature_families=("quality",))
    if policy_id == "quality_only_v1":
        return _PolicySpec(policy_id=policy_id, selector_view=None, selector_feature_families=("quality",))
    if policy_id == "window_support_novelty_v1":
        return _PolicySpec(policy_id=policy_id, selector_view=config.window_view, selector_feature_families=("window",))
    if policy_id == "window_kcenter_v1":
        return _PolicySpec(policy_id=policy_id, selector_view=config.window_view, selector_feature_families=("window",))
    if policy_id == "ts2vec_support_novelty_v1":
        return _PolicySpec(policy_id=policy_id, selector_view=config.ts2vec_view, selector_feature_families=("ts2vec",))
    if policy_id == "ts2vec_kcenter_v1":
        return _PolicySpec(policy_id=policy_id, selector_view=config.ts2vec_view, selector_feature_families=("ts2vec",))
    if policy_id == "submitted_full_replay_v1":
        return _PolicySpec(
            policy_id=policy_id,
            selector_view=None,
            selector_feature_families=("ts2vec", "window", "quality", "artifact"),
        )
    if policy_id == "oracle_greedy_eval_view_v1":
        return _PolicySpec(
            policy_id=policy_id,
            selector_view=None,
            selector_feature_families=("oracle", "target"),
            uses_target_for_selection=True,
        )
    if policy_id == "oracle_greedy_target_family_v1":
        return _PolicySpec(
            policy_id=policy_id,
            selector_view=None,
            selector_feature_families=("oracle", "target_family"),
            uses_target_for_selection=True,
        )
    raise ValueError(f"Unsupported coverage benchmark policy: {policy_id}")


def _rank_candidates(
    policy_spec: _PolicySpec,
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    episode: EpisodeSpec,
    config: CoverageBenchmarkConfig,
    episode_index: int,
    policy_index: int,
) -> tuple[tuple[str, ...], dict[str, float]]:
    eligible = _eligible_candidates(clips_by_id, episode.candidate_ids, config)
    if not eligible:
        return (), {}
    if policy_spec.policy_id == "random_valid_v1":
        rng = _policy_rng(config, episode_index=episode_index, policy_index=policy_index)
        order = tuple(str(sample_id) for sample_id in rng.permutation(eligible))
        return order, {sample_id: float(clips_by_id[sample_id].quality_score) for sample_id in order}
    if policy_spec.policy_id == "quality_stratified_random_v1":
        return _quality_stratified_random_order(
            clips_by_id,
            eligible,
            config=config,
            episode_index=episode_index,
            policy_index=policy_index,
        )
    if policy_spec.policy_id == "quality_only_v1":
        order = _quality_order(clips_by_id, eligible)
        return order, {sample_id: float(clips_by_id[sample_id].quality_score) for sample_id in order}
    if policy_spec.policy_id == "window_support_novelty_v1":
        return _support_novelty_order(
            clips_by_id,
            support_ids=episode.support_ids,
            candidate_ids=eligible,
            representation=str(config.window_view),
            config=config,
        )
    if policy_spec.policy_id == "window_kcenter_v1":
        return _kcenter_order(
            clips_by_id,
            support_ids=episode.support_ids,
            candidate_ids=eligible,
            representation=str(config.window_view),
            config=config,
        )
    if policy_spec.policy_id == "ts2vec_support_novelty_v1":
        return _support_novelty_order(
            clips_by_id,
            support_ids=episode.support_ids,
            candidate_ids=eligible,
            representation=str(config.ts2vec_view),
            config=config,
        )
    if policy_spec.policy_id == "ts2vec_kcenter_v1":
        return _kcenter_order(
            clips_by_id,
            support_ids=episode.support_ids,
            candidate_ids=eligible,
            representation=str(config.ts2vec_view),
            config=config,
        )
    if policy_spec.policy_id == "submitted_full_replay_v1":
        return _submitted_blend_order(
            clips_by_id,
            support_ids=episode.support_ids,
            candidate_ids=eligible,
            config=config,
        )
    if policy_spec.policy_id == "oracle_greedy_eval_view_v1":
        return _oracle_greedy_order(
            clips_by_id,
            episode=episode,
            candidate_ids=eligible,
            config=config,
        )
    if policy_spec.policy_id == "oracle_greedy_target_family_v1":
        return _oracle_target_family_order(
            clips_by_id,
            episode=episode,
            candidate_ids=eligible,
            config=config,
        )
    raise ValueError(f"Unsupported coverage benchmark policy: {policy_spec.policy_id}")


def _eligible_candidates(
    clips_by_id: Mapping[str, BenchmarkClip],
    candidate_ids: Sequence[str],
    config: CoverageBenchmarkConfig,
) -> tuple[str, ...]:
    return tuple(str(sample_id) for sample_id in candidate_ids if _passes_gates(clips_by_id[str(sample_id)], config))


def _passes_gates(clip: BenchmarkClip, config: CoverageBenchmarkConfig) -> bool:
    artifact_ok = config.max_artifact_score is None or float(clip.artifact_score) <= float(config.max_artifact_score)
    return (
        float(clip.quality_score) >= float(config.quality_threshold)
        and float(clip.stationary_fraction) <= float(config.max_stationary_fraction)
        and float(clip.max_abs_value) <= float(config.max_abs_value)
        and artifact_ok
    )


def _policy_rng(config: CoverageBenchmarkConfig, *, episode_index: int, policy_index: int) -> np.random.Generator:
    seed = int(config.random_seed) + int(episode_index) * 1009 + int(policy_index) * 9176
    return np.random.default_rng(seed)


def _quality_order(clips_by_id: Mapping[str, BenchmarkClip], candidate_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            [str(sample_id) for sample_id in candidate_ids],
            key=lambda sample_id: (-float(clips_by_id[sample_id].quality_score), sample_id),
        )
    )


def _quality_stratified_random_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    candidate_ids: Sequence[str],
    *,
    config: CoverageBenchmarkConfig,
    episode_index: int,
    policy_index: int,
) -> tuple[tuple[str, ...], dict[str, float]]:
    quality_order = list(_quality_order(clips_by_id, candidate_ids))
    stratum_size = min(len(quality_order), max(max(int(budget) for budget in config.budgets) * 2, 1))
    stratum = quality_order[:stratum_size]
    remainder = quality_order[stratum_size:]
    rng = _policy_rng(config, episode_index=episode_index, policy_index=policy_index)
    order = [
        *(str(sample_id) for sample_id in rng.permutation(stratum)),
        *(str(sample_id) for sample_id in rng.permutation(remainder)),
    ]
    return tuple(order), {sample_id: float(clips_by_id[sample_id].quality_score) for sample_id in order}


def _support_novelty_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    representation: str,
    config: CoverageBenchmarkConfig,
) -> tuple[tuple[str, ...], dict[str, float]]:
    support = stack_embeddings(clips_by_id, support_ids, representation=representation)
    candidates = stack_embeddings(clips_by_id, candidate_ids, representation=representation)
    distances = _nearest_distances(support, candidates, metric=config.distance_metric)
    scores = {str(sample_id): float(distances[index]) for index, sample_id in enumerate(candidate_ids)}
    order = _score_order(candidate_ids, scores, clips_by_id)
    return order, scores


def _kcenter_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    representation: str,
    config: CoverageBenchmarkConfig,
) -> tuple[tuple[str, ...], dict[str, float]]:
    selected: list[str] = []
    remaining = [str(sample_id) for sample_id in candidate_ids]
    scores: dict[str, float] = {}
    while remaining:
        reference_ids = [*support_ids, *selected]
        reference = stack_embeddings(clips_by_id, reference_ids, representation=representation)
        candidates = stack_embeddings(clips_by_id, remaining, representation=representation)
        distances = _nearest_distances(reference, candidates, metric=config.distance_metric)
        by_id = {sample_id: float(distances[index]) for index, sample_id in enumerate(remaining)}
        best = max(
            remaining,
            key=lambda sample_id: (by_id[sample_id], float(clips_by_id[sample_id].quality_score), sample_id),
        )
        scores[best] = by_id[best]
        selected.append(best)
        remaining.remove(best)
    return tuple(selected), scores


def _submitted_blend_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    config: CoverageBenchmarkConfig,
) -> tuple[tuple[str, ...], dict[str, float]]:
    selected: list[str] = []
    remaining = [str(sample_id) for sample_id in candidate_ids]
    scores: dict[str, float] = {}
    while remaining:
        reference_ids = [*support_ids, *selected]
        left_scores = _distance_scores(
            clips_by_id,
            support_ids=reference_ids,
            candidate_ids=remaining,
            representation=str(config.ts2vec_view),
            config=config,
        )
        right_scores = _distance_scores(
            clips_by_id,
            support_ids=reference_ids,
            candidate_ids=remaining,
            representation=str(config.window_view),
            config=config,
        )
        left_values = np.asarray([left_scores[sample_id] for sample_id in remaining], dtype=float)
        right_values = np.asarray([right_scores[sample_id] for sample_id in remaining], dtype=float)
        blended_values = float(config.blend_alpha) * _minmax(left_values) + (1.0 - float(config.blend_alpha)) * _minmax(
            right_values
        )
        blended = {sample_id: float(blended_values[index]) for index, sample_id in enumerate(remaining)}
        best = max(
            remaining,
            key=lambda sample_id: (blended[sample_id], float(clips_by_id[sample_id].quality_score), sample_id),
        )
        scores[best] = blended[best]
        selected.append(best)
        remaining.remove(best)
    return tuple(selected), scores


def _distance_scores(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    support_ids: Sequence[str],
    candidate_ids: Sequence[str],
    representation: str,
    config: CoverageBenchmarkConfig,
) -> dict[str, float]:
    support = stack_embeddings(clips_by_id, support_ids, representation=representation)
    candidates = stack_embeddings(clips_by_id, candidate_ids, representation=representation)
    distances = _nearest_distances(support, candidates, metric=config.distance_metric)
    return {str(sample_id): float(distances[index]) for index, sample_id in enumerate(candidate_ids)}


def _oracle_greedy_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    episode: EpisodeSpec,
    candidate_ids: Sequence[str],
    config: CoverageBenchmarkConfig,
) -> tuple[tuple[str, ...], dict[str, float]]:
    selected: list[str] = []
    remaining = [str(sample_id) for sample_id in candidate_ids]
    scores: dict[str, float] = {}
    max_budget = min(max(int(budget) for budget in config.budgets), len(remaining))
    while remaining and len(selected) < max_budget:
        by_id: dict[str, float] = {}
        for sample_id in remaining:
            trial = [*selected, sample_id]
            by_id[sample_id] = _mean_primary_coverage_gain(
                clips_by_id,
                episode=episode,
                selected_ids=trial,
                config=config,
            )
        best = max(remaining, key=lambda sample_id: (by_id[sample_id], float(clips_by_id[sample_id].quality_score), sample_id))
        scores[best] = by_id[best]
        selected.append(best)
        remaining.remove(best)
    selected_set = set(selected)
    tail = [sample_id for sample_id in remaining if sample_id not in selected_set]
    for sample_id in tail:
        scores[sample_id] = 0.0
    return tuple([*selected, *tail]), scores


def _oracle_target_family_order(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    episode: EpisodeSpec,
    candidate_ids: Sequence[str],
    config: CoverageBenchmarkConfig,
) -> tuple[tuple[str, ...], dict[str, float]]:
    labels_by_id = _source_family_labels_by_id(
        clips_by_id,
        representation=str(config.source_family_label_view),
        source_family_count=int(config.source_family_count),
    )
    target_labels = {labels_by_id[str(sample_id)] for sample_id in episode.target_ids if str(sample_id) in labels_by_id}
    known_labels = {labels_by_id[str(sample_id)] for sample_id in episode.support_ids if str(sample_id) in labels_by_id}
    selected: list[str] = []
    remaining = [str(sample_id) for sample_id in candidate_ids]
    scores: dict[str, float] = {}
    max_budget = min(max(int(budget) for budget in config.budgets), len(remaining))
    while remaining and len(selected) < max_budget:
        by_id: dict[str, float] = {}
        selected_labels = {labels_by_id[str(sample_id)] for sample_id in selected if str(sample_id) in labels_by_id}
        for sample_id in remaining:
            trial_labels = set(known_labels)
            trial_labels.update(selected_labels)
            if sample_id in labels_by_id:
                trial_labels.add(labels_by_id[sample_id])
            discovered = target_labels & trial_labels
            by_id[sample_id] = float(len(discovered) / len(target_labels)) if target_labels else 0.0
        best = max(remaining, key=lambda sample_id: (by_id[sample_id], float(clips_by_id[sample_id].quality_score), sample_id))
        scores[best] = by_id[best]
        selected.append(best)
        remaining.remove(best)
    selected_set = set(selected)
    tail = [sample_id for sample_id in remaining if sample_id not in selected_set]
    for sample_id in tail:
        scores[sample_id] = 0.0
    return tuple([*selected, *tail]), scores


def _source_family_labels_by_id(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    representation: str,
    source_family_count: int,
) -> dict[str, str]:
    clips = [clips_by_id[sample_id] for sample_id in sorted(clips_by_id)]
    family_by_group = infer_source_family_assignments(
        clips,
        representation=representation,
        source_family_count=source_family_count,
    )
    return {clip.sample_id: str(family_by_group[str(clip.source_group_id)]) for clip in clips}


def _mean_primary_coverage_gain(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    episode: EpisodeSpec,
    selected_ids: Sequence[str],
    config: CoverageBenchmarkConfig,
) -> float:
    gains = []
    for view in config.primary_eval_views:
        support = stack_embeddings(clips_by_id, episode.support_ids, representation=str(view))
        target = stack_embeddings(clips_by_id, episode.target_ids, representation=str(view))
        selected = stack_embeddings(clips_by_id, selected_ids, representation=str(view)) if selected_ids else np.empty((0, 0))
        _d0, _dk, rel = _coverage_distance_arrays(support, target, selected, config=config)
        gains.append(float(np.mean(rel)) if len(rel) else 0.0)
    return float(np.mean(gains)) if gains else 0.0


def _coverage_metric_rows(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    episode: EpisodeSpec,
    selected_ids: Sequence[str],
    policy_spec: _PolicySpec,
    config: CoverageBenchmarkConfig,
    budget_k: int,
) -> list[CoverageMetricRow]:
    rows: list[CoverageMetricRow] = []
    for eval_view in config.eval_views:
        eval_view = str(eval_view)
        support = stack_embeddings(clips_by_id, episode.support_ids, representation=eval_view)
        target = stack_embeddings(clips_by_id, episode.target_ids, representation=eval_view)
        selected = stack_embeddings(clips_by_id, selected_ids, representation=eval_view) if selected_ids else np.empty((0, 0))
        baseline_distances, after_distances, relative_gains = _coverage_distance_arrays(
            support,
            target,
            selected,
            config=config,
        )
        absolute_gains = baseline_distances - after_distances
        tau = _support_tau(support, metric=config.distance_metric)
        tail_mask = baseline_distances >= float(np.quantile(baseline_distances, 0.90)) if len(baseline_distances) else np.asarray([])
        tail_gain = float(np.mean(relative_gains[tail_mask])) if np.any(tail_mask) else 0.0
        selector_overlap = _selector_feature_overlap(eval_view, policy_spec, config)
        primary_eval = (
            eval_view in set(str(view) for view in config.primary_eval_views)
            and not selector_overlap
            and not policy_spec.uses_target_for_selection
        )
        rows.extend(
            [
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "coverage_gain_rel",
                    _mean(relative_gains),
                    higher_is_better=True,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "coverage_gain_abs",
                    _mean(absolute_gains),
                    higher_is_better=True,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "mean_nn_before",
                    _mean(baseline_distances),
                    higher_is_better=False,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "mean_nn_after",
                    _mean(after_distances),
                    higher_is_better=False,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "tau_coverage_before",
                    float(np.mean(baseline_distances <= tau)) if len(baseline_distances) else 0.0,
                    higher_is_better=True,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "tau_coverage_after",
                    float(np.mean(after_distances <= tau)) if len(after_distances) else 0.0,
                    higher_is_better=True,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "tau_coverage_gain",
                    (
                        float(np.mean(after_distances <= tau) - np.mean(baseline_distances <= tau))
                        if len(after_distances) and len(baseline_distances)
                        else 0.0
                    ),
                    higher_is_better=True,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "tail_coverage_gain_rel_q90",
                    tail_gain,
                    higher_is_better=True,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
                _metric_row(
                    episode,
                    policy_spec,
                    budget_k,
                    eval_view,
                    "selected_support_duplicate_rate",
                    _selected_support_duplicate_rate(support, selected, metric=config.distance_metric),
                    higher_is_better=False,
                    primary_eval=primary_eval,
                    selector_feature_overlap=selector_overlap,
                ),
            ]
        )
    return rows


def _coverage_distance_arrays(
    support: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
    *,
    config: CoverageBenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.asarray(support, dtype=float)
    target = np.asarray(target, dtype=float)
    selected = np.asarray(selected, dtype=float)
    baseline = _nearest_distances(support, target, metric=config.distance_metric)
    if selected.size:
        augmented = np.vstack([support, selected])
    else:
        augmented = support
    after = _nearest_distances(augmented, target, metric=config.distance_metric)
    denominator = np.maximum(baseline, float(config.eps))
    relative = np.maximum((baseline - after) / denominator, 0.0)
    return baseline, after, relative


def _hygiene_metric_rows(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    episode: EpisodeSpec,
    selected_ids: Sequence[str],
    policy_spec: _PolicySpec,
    config: CoverageBenchmarkConfig,
    budget_k: int,
) -> list[CoverageMetricRow]:
    selected = [str(sample_id) for sample_id in selected_ids]
    selected_count = len(selected)
    candidate_set = set(str(sample_id) for sample_id in episode.candidate_ids)
    target_set = set(str(sample_id) for sample_id in episode.target_ids)
    invalid_count = sum(not _passes_gates(clips_by_id[sample_id], config) for sample_id in selected)
    duplicate_count = selected_count - len(set(selected))
    metrics = {
        "selected_count": float(selected_count),
        "selected_invalid_rate": float(invalid_count / selected_count) if selected_count else 0.0,
        "selected_out_of_pool_count": float(sum(sample_id not in candidate_set for sample_id in selected)),
        "selected_target_leak_count": float(sum(sample_id in target_set for sample_id in selected)),
        "selected_duplicate_clip_count": float(duplicate_count),
    }
    return [
        _metric_row(
            episode,
            policy_spec,
            budget_k,
            HYGIENE_EVAL_VIEW,
            metric_name,
            metric_value,
            higher_is_better=metric_name == "selected_count",
            primary_eval=True,
            selector_feature_overlap=False,
        )
        for metric_name, metric_value in metrics.items()
    ]


def _selected_rows(
    clips_by_id: Mapping[str, BenchmarkClip],
    *,
    episode: EpisodeSpec,
    selected_ids: Sequence[str],
    scores: Mapping[str, float],
    policy_spec: _PolicySpec,
    config: CoverageBenchmarkConfig,
    budget_k: int,
) -> list[CoverageSelectedRow]:
    rows = []
    for rank_index, sample_id in enumerate(selected_ids, start=1):
        clip = clips_by_id[str(sample_id)]
        rows.append(
            CoverageSelectedRow(
                episode_id=episode.episode_id,
                fold_id=int(episode.fold_id),
                policy_id=policy_spec.policy_id,
                budget_k=int(budget_k),
                rank_index=int(rank_index),
                sample_id=str(sample_id),
                source_group_id=str(clip.source_group_id),
                score=float(scores.get(str(sample_id), 0.0)),
                quality_score=float(clip.quality_score),
                artifact_score=float(clip.artifact_score),
                valid=_passes_gates(clip, config),
                passed_artifact_gate=(
                    config.max_artifact_score is None or float(clip.artifact_score) <= float(config.max_artifact_score)
                ),
            )
        )
    return rows


def _metric_row(
    episode: EpisodeSpec,
    policy_spec: _PolicySpec,
    budget_k: int,
    eval_view: str,
    metric_name: str,
    metric_value: float,
    *,
    higher_is_better: bool,
    primary_eval: bool,
    selector_feature_overlap: bool,
) -> CoverageMetricRow:
    return CoverageMetricRow(
        episode_id=episode.episode_id,
        fold_id=int(episode.fold_id),
        policy_id=policy_spec.policy_id,
        budget_k=int(budget_k),
        eval_view=str(eval_view),
        metric_name=str(metric_name),
        metric_value=float(metric_value),
        higher_is_better=bool(higher_is_better),
        primary_eval=bool(primary_eval),
        selector_feature_overlap=bool(selector_feature_overlap),
        uses_target_for_selection=bool(policy_spec.uses_target_for_selection),
    )


def _primary_gain(
    metric_rows: Sequence[CoverageMetricRow],
    episode_id: str,
    policy_id: str,
    budget_k: int,
) -> float:
    values = [
        float(row.metric_value)
        for row in metric_rows
        if row.episode_id == episode_id
        and row.policy_id == policy_id
        and int(row.budget_k) == int(budget_k)
        and row.metric_name == "coverage_gain_rel"
        and row.primary_eval
    ]
    return float(np.mean(values)) if values else 0.0


def _policy_summary(
    metric_rows: Sequence[CoverageMetricRow],
    round_rows: Sequence[CoverageRoundRow],
) -> dict[str, dict[str, float]]:
    by_policy: dict[str, list[float]] = {}
    diagnostic_by_policy: dict[str, list[float]] = {}
    for row in metric_rows:
        if row.metric_name != "coverage_gain_rel":
            continue
        if row.primary_eval:
            by_policy.setdefault(row.policy_id, []).append(float(row.metric_value))
        else:
            diagnostic_by_policy.setdefault(row.policy_id, []).append(float(row.metric_value))
    round_count_by_policy: dict[str, int] = {}
    underfilled_by_policy: dict[str, int] = {}
    for row in round_rows:
        round_count_by_policy[row.policy_id] = round_count_by_policy.get(row.policy_id, 0) + 1
        if row.status != "ok":
            underfilled_by_policy[row.policy_id] = underfilled_by_policy.get(row.policy_id, 0) + 1
    policy_ids = sorted(set(round_count_by_policy) | set(by_policy) | set(diagnostic_by_policy))
    return {
        policy_id: {
            "mean_primary_coverage_gain_rel": float(np.mean(by_policy.get(policy_id, [0.0]))),
            "mean_diagnostic_coverage_gain_rel": float(np.mean(diagnostic_by_policy.get(policy_id, [0.0]))),
            "primary_row_count": float(len(by_policy.get(policy_id, []))),
            "diagnostic_row_count": float(len(diagnostic_by_policy.get(policy_id, []))),
            "round_count": float(round_count_by_policy.get(policy_id, 0)),
            "underfilled_round_count": float(underfilled_by_policy.get(policy_id, 0)),
        }
        for policy_id in policy_ids
    }


def _score_order(
    candidate_ids: Sequence[str],
    scores: Mapping[str, float],
    clips_by_id: Mapping[str, BenchmarkClip],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            [str(sample_id) for sample_id in candidate_ids],
            key=lambda sample_id: (
                -float(scores[sample_id]),
                -float(clips_by_id[sample_id].quality_score),
                sample_id,
            ),
        )
    )


def _selector_feature_overlap(eval_view: str, policy_spec: _PolicySpec, config: CoverageBenchmarkConfig) -> bool:
    eval_family = _feature_family(eval_view, config)
    return eval_family in set(policy_spec.selector_feature_families)


def _feature_family(view: str, config: CoverageBenchmarkConfig) -> str:
    view = str(view)
    if view in config.eval_view_families:
        return str(config.eval_view_families[view])
    lowered = view.lower()
    if lowered.startswith("ts2vec"):
        return "ts2vec"
    if lowered.startswith("window"):
        return "window"
    if lowered.startswith("morph"):
        return "morphology"
    if lowered.startswith("raw_shape"):
        return "raw_shape"
    return lowered


def _nearest_distances(reference_embeddings: np.ndarray, query_embeddings: np.ndarray, *, metric: str) -> np.ndarray:
    reference = np.asarray(reference_embeddings, dtype=float)
    query = np.asarray(query_embeddings, dtype=float)
    if len(query) == 0:
        return np.zeros((0,), dtype=float)
    if len(reference) == 0:
        return np.full((len(query),), np.inf, dtype=float)
    distances = _pairwise_distances(query, reference, metric=metric)
    return np.min(distances, axis=1)


def _pairwise_distances(query_embeddings: np.ndarray, reference_embeddings: np.ndarray, *, metric: str) -> np.ndarray:
    query = np.asarray(query_embeddings, dtype=float)
    reference = np.asarray(reference_embeddings, dtype=float)
    if metric == "euclidean":
        delta = query[:, None, :] - reference[None, :, :]
        return np.linalg.norm(delta, axis=2)
    if metric == "cosine":
        query_norm = normalize_rows(query)
        reference_norm = normalize_rows(reference)
        similarity = query_norm @ reference_norm.T
        return np.maximum(1.0 - similarity, 0.0)
    raise ValueError(f"Unsupported coverage distance metric: {metric}")


def _support_tau(support_embeddings: np.ndarray, *, metric: str) -> float:
    support = np.asarray(support_embeddings, dtype=float)
    if len(support) < 2:
        return 0.0
    distances = _pairwise_distances(support, support, metric=metric)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if len(finite) == 0:
        return 0.0
    return float(np.quantile(finite, 0.90))


def _selected_support_duplicate_rate(support_embeddings: np.ndarray, selected_embeddings: np.ndarray, *, metric: str) -> float:
    selected = np.asarray(selected_embeddings, dtype=float)
    if selected.size == 0 or len(selected) == 0:
        return 0.0
    tau = _support_tau(support_embeddings, metric=metric)
    distances = _nearest_distances(support_embeddings, selected, metric=metric)
    return float(np.mean(distances <= tau))


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1.0e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if values.size else 0.0


def _emit_progress(callback: Callable[[dict[str, object]], None] | None, event: str, **payload: object) -> None:
    if callback is not None:
        callback({"event": event, **payload})
