from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BenchmarkClip:
    sample_id: str
    source_group_id: str
    embeddings: Mapping[str, np.ndarray]
    quality_score: float = 1.0
    stationary_fraction: float = 0.0
    max_abs_value: float = 0.0
    artifact_score: float = 0.0


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    fold_id: int
    support_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    support_group_ids: tuple[str, ...]
    candidate_group_ids: tuple[str, ...]
    target_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class OfflineBenchmarkConfig:
    batch_size: int
    rounds: int
    policies: Sequence[str]
    representations: Sequence[str]
    primary_representations: Sequence[str]
    random_seed: int = 17
    quality_threshold: float = 0.85
    max_stationary_fraction: float = 0.90
    max_abs_value: float = 60.0
    oracle_candidate_cap: int | None = None
    oracle_exact_combination_limit: int = 10_000
    blend_left_representation: str = "ts2vec"
    blend_right_representation: str = "window"
    blend_alpha: float = 0.5
    max_artifact_score: float | None = 0.05


@dataclass
class RoundResult:
    episode_id: str
    fold_id: int
    policy_name: str
    round_index: int
    batch_size: int
    support_ids_before: tuple[str, ...]
    candidate_ids_before: tuple[str, ...]
    selected_ids: tuple[str, ...]
    selected_scores: tuple[float, ...]
    selected_source_group_ids: tuple[str, ...]
    support_ids_after: tuple[str, ...]
    candidate_ids_after: tuple[str, ...]
    coverage_by_representation: dict[str, dict[str, float]]
    balanced_relative_gain: float
    cumulative_balanced_relative_gain: float
    oracle_fraction: float = 0.0
    selection_summary: dict[str, float | int] = field(default_factory=dict)
    selection_details: list[dict[str, object]] = field(default_factory=list)

    @property
    def support_count_before(self) -> int:
        return len(self.support_ids_before)

    @property
    def support_count_after(self) -> int:
        return len(self.support_ids_after)

    @property
    def candidate_count_before(self) -> int:
        return len(self.candidate_ids_before)

    @property
    def candidate_count_after(self) -> int:
        return len(self.candidate_ids_after)


@dataclass(frozen=True)
class BenchmarkResult:
    episodes: tuple[EpisodeSpec, ...]
    policies: tuple[str, ...]
    rounds: tuple[RoundResult, ...]
    policy_summary: dict[str, dict[str, float]]
    difficulty_audit: tuple[dict[str, object], ...]
    config: OfflineBenchmarkConfig
