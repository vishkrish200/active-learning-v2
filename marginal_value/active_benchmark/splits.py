from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from marginal_value.active_benchmark.schema import BenchmarkClip, EpisodeSpec
from marginal_value.indexing.knn_features import normalize_rows


def build_source_blocked_episodes(
    clips: Sequence[BenchmarkClip],
    *,
    n_folds: int,
    candidate_groups_per_episode: int,
    target_groups_per_episode: int,
    max_support_groups: int | None = None,
) -> tuple[EpisodeSpec, ...]:
    by_group: dict[str, list[BenchmarkClip]] = defaultdict(list)
    for clip in clips:
        by_group[str(clip.source_group_id)].append(clip)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("Cannot build active benchmark episodes without clips.")
    required = int(candidate_groups_per_episode) + int(target_groups_per_episode) + 1
    if len(groups) < required:
        raise ValueError("Not enough source groups for blocked support/candidate/target roles.")

    episodes: list[EpisodeSpec] = []
    target_count = max(1, int(target_groups_per_episode))
    candidate_count = max(1, int(candidate_groups_per_episode))
    for fold_id in range(max(1, int(n_folds))):
        rotated = groups[fold_id * target_count :] + groups[: fold_id * target_count]
        target_groups = tuple(rotated[:target_count])
        candidate_groups = tuple(rotated[target_count : target_count + candidate_count])
        support_groups = [
            group
            for group in rotated[target_count + candidate_count :]
            if group not in set(target_groups) and group not in set(candidate_groups)
        ]
        if max_support_groups is not None:
            support_groups = support_groups[: max(1, int(max_support_groups))]
        if not support_groups:
            raise ValueError("Each active benchmark episode requires at least one support group.")
        episodes.append(
            EpisodeSpec(
                episode_id=f"episode_{fold_id:03d}",
                fold_id=int(fold_id),
                support_ids=_clip_ids_for_groups(by_group, support_groups),
                candidate_ids=_clip_ids_for_groups(by_group, candidate_groups),
                target_ids=_clip_ids_for_groups(by_group, target_groups),
                support_group_ids=tuple(support_groups),
                candidate_group_ids=candidate_groups,
                target_group_ids=target_groups,
            )
        )
    return tuple(episodes)


def build_difficulty_targeted_episodes(
    clips: Sequence[BenchmarkClip],
    *,
    n_folds: int,
    candidate_groups_per_episode: int,
    target_groups_per_episode: int,
    max_support_groups: int | None = None,
    representation: str = "window",
) -> tuple[EpisodeSpec, ...]:
    by_group: dict[str, list[BenchmarkClip]] = defaultdict(list)
    for clip in clips:
        by_group[str(clip.source_group_id)].append(clip)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("Cannot build active benchmark episodes without clips.")
    target_count = max(1, int(target_groups_per_episode))
    candidate_count = max(1, int(candidate_groups_per_episode))
    required = target_count + candidate_count + 1
    if len(groups) < required:
        raise ValueError("Not enough source groups for blocked support/candidate/target roles.")

    centroids = _group_centroids(by_group, groups, representation=representation)
    distances = _cosine_distance_matrix(centroids, centroids)
    target_order = _target_order_by_isolation(groups, distances)
    episodes: list[EpisodeSpec] = []
    for fold_id in range(max(1, int(n_folds))):
        target_groups = tuple(_rotating_take(target_order, start=fold_id * target_count, count=target_count))
        target_indices = [groups.index(group) for group in target_groups]
        target_distance = distances[:, target_indices].mean(axis=1)
        candidate_groups = tuple(
            group
            for group in sorted(
                [group for group in groups if group not in set(target_groups)],
                key=lambda group: (float(target_distance[groups.index(group)]), group),
            )[:candidate_count]
        )
        blocked = set(target_groups) | set(candidate_groups)
        support_pool = [group for group in groups if group not in blocked]
        support_groups = sorted(
            support_pool,
            key=lambda group: (-float(target_distance[groups.index(group)]), group),
        )
        if max_support_groups is not None:
            support_groups = support_groups[: max(1, int(max_support_groups))]
        if not support_groups:
            raise ValueError("Each active benchmark episode requires at least one support group.")
        episodes.append(
            EpisodeSpec(
                episode_id=f"episode_{fold_id:03d}",
                fold_id=int(fold_id),
                support_ids=_clip_ids_for_groups(by_group, support_groups),
                candidate_ids=_clip_ids_for_groups(by_group, candidate_groups),
                target_ids=_clip_ids_for_groups(by_group, target_groups),
                support_group_ids=tuple(support_groups),
                candidate_group_ids=candidate_groups,
                target_group_ids=target_groups,
            )
        )
    return tuple(episodes)


def build_opportunity_targeted_episodes(
    clips: Sequence[BenchmarkClip],
    *,
    n_folds: int,
    candidate_groups_per_episode: int,
    target_groups_per_episode: int,
    max_support_groups: int | None = None,
    representation: str = "window",
) -> tuple[EpisodeSpec, ...]:
    by_group: dict[str, list[BenchmarkClip]] = defaultdict(list)
    for clip in clips:
        by_group[str(clip.source_group_id)].append(clip)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("Cannot build active benchmark episodes without clips.")
    target_count = max(1, int(target_groups_per_episode))
    candidate_count = max(1, int(candidate_groups_per_episode))
    required = target_count + candidate_count + 1
    if len(groups) < required:
        raise ValueError("Not enough source groups for blocked support/candidate/target roles.")

    centroids = _group_centroids(by_group, groups, representation=representation)
    distances = _cosine_distance_matrix(centroids, centroids)
    target_order = _target_order_by_isolation(groups, distances)
    episodes: list[EpisodeSpec] = []
    for fold_id in range(max(1, int(n_folds))):
        target_groups = tuple(_rotating_take(target_order, start=fold_id * target_count, count=target_count))
        target_indices = [groups.index(group) for group in target_groups]
        target_distance = distances[:, target_indices].mean(axis=1)
        candidate_groups = _opportunity_candidate_groups(
            groups,
            target_groups=target_groups,
            target_distance=target_distance,
            candidate_count=candidate_count,
        )
        blocked = set(target_groups) | set(candidate_groups)
        support_pool = [group for group in groups if group not in blocked]
        support_groups = sorted(
            support_pool,
            key=lambda group: (-float(target_distance[groups.index(group)]), group),
        )
        if max_support_groups is not None:
            support_groups = support_groups[: max(1, int(max_support_groups))]
        if not support_groups:
            raise ValueError("Each active benchmark episode requires at least one support group.")
        episodes.append(
            EpisodeSpec(
                episode_id=f"episode_{fold_id:03d}",
                fold_id=int(fold_id),
                support_ids=_clip_ids_for_groups(by_group, support_groups),
                candidate_ids=_clip_ids_for_groups(by_group, candidate_groups),
                target_ids=_clip_ids_for_groups(by_group, target_groups),
                support_group_ids=tuple(support_groups),
                candidate_group_ids=candidate_groups,
                target_group_ids=target_groups,
            )
        )
    return tuple(episodes)


def _clip_ids_for_groups(by_group: dict[str, list[BenchmarkClip]], groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        clip.sample_id
        for group in groups
        for clip in sorted(by_group[str(group)], key=lambda row: row.sample_id)
    )


def _group_centroids(
    by_group: dict[str, list[BenchmarkClip]],
    groups: Sequence[str],
    *,
    representation: str,
) -> np.ndarray:
    centroids: list[np.ndarray] = []
    for group in groups:
        embeddings = [
            np.asarray(clip.embeddings[representation], dtype=float).reshape(-1)
            for clip in by_group[str(group)]
        ]
        if not embeddings:
            raise ValueError(f"Source group {group} has no clips.")
        centroids.append(np.mean(np.vstack(embeddings), axis=0))
    return np.vstack(centroids)


def _cosine_distance_matrix(left_embeddings: np.ndarray, right_embeddings: np.ndarray) -> np.ndarray:
    left = normalize_rows(np.asarray(left_embeddings, dtype=float))
    right = normalize_rows(np.asarray(right_embeddings, dtype=float))
    return np.maximum(1.0 - left @ right.T, 0.0)


def _target_order_by_isolation(groups: Sequence[str], distances: np.ndarray) -> list[str]:
    isolation_scores = []
    for index, group in enumerate(groups):
        row = np.delete(distances[index], index)
        isolation_scores.append((float(np.mean(row)) if len(row) else 0.0, group))
    return [group for _score, group in sorted(isolation_scores, key=lambda row: (-row[0], row[1]))]


def _opportunity_candidate_groups(
    groups: Sequence[str],
    *,
    target_groups: Sequence[str],
    target_distance: np.ndarray,
    candidate_count: int,
) -> tuple[str, ...]:
    target_set = {str(group) for group in target_groups}
    available = [str(group) for group in groups if str(group) not in target_set]
    if len(available) < int(candidate_count):
        raise ValueError("Not enough non-target groups to build opportunity-targeted candidates.")
    near_count = max(1, int(candidate_count) // 2)
    far_count = max(0, int(candidate_count) - near_count)
    near_groups = [
        group
        for group in sorted(
            available,
            key=lambda group: (float(target_distance[groups.index(group)]), group),
        )[:near_count]
    ]
    near_set = set(near_groups)
    far_groups = [
        group
        for group in sorted(
            [group for group in available if group not in near_set],
            key=lambda group: (-float(target_distance[groups.index(group)]), group),
        )[:far_count]
    ]
    selected = [*near_groups, *far_groups]
    if len(selected) < int(candidate_count):
        selected.extend(
            group
            for group in sorted(available, key=lambda group: (float(target_distance[groups.index(group)]), group))
            if group not in set(selected)
        )
    return tuple(selected[: int(candidate_count)])


def _rotating_take(values: Sequence[str], *, start: int, count: int) -> list[str]:
    if not values:
        return []
    return [values[(start + offset) % len(values)] for offset in range(count)]
