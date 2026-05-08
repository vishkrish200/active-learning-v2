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


def build_source_family_shift_episodes(
    clips: Sequence[BenchmarkClip],
    *,
    n_folds: int,
    candidate_groups_per_episode: int,
    target_groups_per_episode: int,
    max_support_groups: int | None = None,
    representation: str = "window",
    source_family_count: int = 4,
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
    family_by_group, groups_by_family, target_family_order = _source_family_roles(
        groups,
        centroids=centroids,
        source_family_count=source_family_count,
        target_count=target_count,
        candidate_count=candidate_count,
    )
    if len(target_family_order) < 2:
        raise ValueError("Source-family shift episodes require at least two source families with target groups.")

    episodes: list[EpisodeSpec] = []
    for fold_id in range(max(1, int(n_folds))):
        target_family = target_family_order[fold_id % len(target_family_order)]
        target_pool = sorted(groups_by_family[target_family])
        target_groups = tuple(_rotating_take(target_pool, start=(fold_id // len(target_family_order)) * target_count, count=target_count))
        target_indices = [groups.index(group) for group in target_groups]
        target_distance = distances[:, target_indices].mean(axis=1)
        non_target_family_groups = [group for group in groups if family_by_group[group] != target_family]
        candidate_groups = _opportunity_candidate_groups(
            non_target_family_groups,
            target_groups=(),
            target_distance=np.asarray([target_distance[groups.index(group)] for group in non_target_family_groups], dtype=float),
            candidate_count=candidate_count,
        )
        blocked = set(target_groups) | set(candidate_groups)
        support_pool = [group for group in non_target_family_groups if group not in blocked]
        support_groups = sorted(
            support_pool,
            key=lambda group: (-float(target_distance[groups.index(group)]), family_by_group[group], group),
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
                candidate_group_ids=tuple(candidate_groups),
                target_group_ids=target_groups,
            )
        )
    return tuple(episodes)


def build_source_family_label_holdout_episodes(
    clips: Sequence[BenchmarkClip],
    *,
    n_folds: int,
    candidate_groups_per_episode: int,
    target_groups_per_episode: int,
    target_candidate_groups_per_episode: int | None = None,
    target_families_per_episode: int = 1,
    max_support_groups: int | None = None,
    representation: str = "window",
    source_family_count: int = 4,
) -> tuple[EpisodeSpec, ...]:
    by_group: dict[str, list[BenchmarkClip]] = defaultdict(list)
    for clip in clips:
        by_group[str(clip.source_group_id)].append(clip)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("Cannot build active benchmark episodes without clips.")
    target_count = max(1, int(target_groups_per_episode))
    candidate_count = max(1, int(candidate_groups_per_episode))
    target_candidate_count = (
        max(1, int(target_candidate_groups_per_episode))
        if target_candidate_groups_per_episode is not None
        else max(1, candidate_count // 2)
    )
    target_candidate_count = min(target_candidate_count, candidate_count)
    decoy_candidate_count = candidate_count - target_candidate_count
    target_family_count = max(1, int(target_families_per_episode))
    if target_count < target_family_count or target_candidate_count < target_family_count:
        raise ValueError("Source-family label-holdout episodes require at least one target and bridge candidate group per target family.")
    target_counts_by_family = _balanced_counts(target_count, target_family_count)
    target_candidate_counts_by_family = _balanced_counts(target_candidate_count, target_family_count)

    centroids = _group_centroids(by_group, groups, representation=representation)
    distances = _cosine_distance_matrix(centroids, centroids)
    family_by_group, groups_by_family, _target_family_order = _source_family_roles(
        groups,
        centroids=centroids,
        source_family_count=source_family_count,
        target_count=target_count,
        candidate_count=candidate_count,
    )
    eligible_families = _eligible_label_holdout_families(
        groups,
        groups_by_family,
        target_count=max(target_counts_by_family),
        target_candidate_count=max(target_candidate_counts_by_family),
        decoy_candidate_count=decoy_candidate_count,
    )
    if len(eligible_families) < target_family_count:
        raise ValueError("Source-family label-holdout episodes require at least one family with target and candidate groups.")

    episodes: list[EpisodeSpec] = []
    for fold_id in range(max(1, int(n_folds))):
        target_families = tuple(
            _rotating_take(
                eligible_families,
                start=fold_id * target_family_count,
                count=target_family_count,
            )
        )
        target_groups_list: list[str] = []
        target_candidate_groups_list: list[str] = []
        family_cycle = fold_id // max(1, len(eligible_families))
        for family_index, target_family in enumerate(target_families):
            target_pool = sorted(groups_by_family[target_family])
            family_target_count = target_counts_by_family[family_index]
            family_candidate_count = target_candidate_counts_by_family[family_index]
            family_cycle_start = family_cycle * (family_target_count + family_candidate_count)
            family_target_groups = _rotating_take(
                target_pool,
                start=family_cycle_start,
                count=family_target_count,
            )
            target_groups_list.extend(family_target_groups)
            target_group_set = set(family_target_groups)
            target_candidate_pool = [group for group in target_pool if group not in target_group_set]
            target_candidate_groups_list.extend(
                _rotating_take(
                    target_candidate_pool,
                    start=family_cycle_start,
                    count=family_candidate_count,
                )
            )
        target_groups = tuple(target_groups_list)
        target_candidate_groups = tuple(target_candidate_groups_list)

        target_indices = [groups.index(group) for group in target_groups]
        target_distance = distances[:, target_indices].mean(axis=1)
        target_family_set = set(target_families)
        non_target_family_groups = [group for group in groups if family_by_group[group] not in target_family_set]
        if decoy_candidate_count > 0:
            decoy_target_distance = np.asarray([target_distance[groups.index(group)] for group in non_target_family_groups], dtype=float)
            decoy_candidate_groups = _opportunity_candidate_groups(
                non_target_family_groups,
                target_groups=(),
                target_distance=decoy_target_distance,
                candidate_count=decoy_candidate_count,
            )
        else:
            decoy_candidate_groups = ()
        candidate_groups = tuple([*target_candidate_groups, *decoy_candidate_groups])
        blocked = set(target_groups) | set(candidate_groups)
        support_pool = [group for group in non_target_family_groups if group not in blocked]
        support_groups = sorted(
            support_pool,
            key=lambda group: (-float(target_distance[groups.index(group)]), family_by_group[group], group),
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
                candidate_group_ids=tuple(candidate_groups),
                target_group_ids=target_groups,
            )
        )
    return tuple(episodes)


def infer_source_family_assignments(
    clips: Sequence[BenchmarkClip],
    *,
    representation: str = "window",
    source_family_count: int = 4,
) -> dict[str, str]:
    by_group: dict[str, list[BenchmarkClip]] = defaultdict(list)
    for clip in clips:
        by_group[str(clip.source_group_id)].append(clip)
    groups = sorted(by_group)
    if not groups:
        raise ValueError("Cannot infer source families without clips.")
    literal = _literal_source_family_assignments(groups)
    if literal is not None:
        return literal
    centroids = _group_centroids(by_group, groups, representation=representation)
    return _cluster_source_family_assignments(
        centroids,
        groups,
        source_family_count=max(2, int(source_family_count)),
    )


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


def _source_family_roles(
    groups: Sequence[str],
    *,
    centroids: np.ndarray,
    source_family_count: int,
    target_count: int,
    candidate_count: int,
) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    literal = _literal_source_family_assignments(groups)
    if literal is not None:
        groups_by_family = _groups_by_family(groups, literal)
        return literal, groups_by_family, _eligible_target_families(
            groups,
            groups_by_family,
            target_count=target_count,
            candidate_count=candidate_count,
        )

    max_family_count = min(len(groups), max(int(source_family_count), 2) + 12)
    last_family_by_group: dict[str, str] | None = None
    last_groups_by_family: dict[str, list[str]] | None = None
    last_target_family_order: list[str] = []
    for family_count in range(max(2, int(source_family_count)), max_family_count + 1):
        family_by_group = _cluster_source_family_assignments(
            centroids,
            groups,
            source_family_count=family_count,
        )
        groups_by_family = _groups_by_family(groups, family_by_group)
        target_family_order = _eligible_target_families(
            groups,
            groups_by_family,
            target_count=target_count,
            candidate_count=candidate_count,
        )
        last_family_by_group = family_by_group
        last_groups_by_family = groups_by_family
        last_target_family_order = target_family_order
        if len(target_family_order) >= 2:
            break
    if last_family_by_group is None or last_groups_by_family is None:
        raise ValueError("Cannot infer source-family assignments without source groups.")
    return last_family_by_group, last_groups_by_family, last_target_family_order


def _groups_by_family(groups: Sequence[str], family_by_group: dict[str, str]) -> dict[str, list[str]]:
    groups_by_family: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        groups_by_family[family_by_group[str(group)]].append(str(group))
    return groups_by_family


def _eligible_target_families(
    groups: Sequence[str],
    groups_by_family: dict[str, list[str]],
    *,
    target_count: int,
    candidate_count: int,
) -> list[str]:
    return sorted(
        [
            family
            for family, family_groups in groups_by_family.items()
            if len(family_groups) >= int(target_count)
            and len(groups) - len(family_groups) >= int(candidate_count) + 1
        ],
        key=lambda family: (family, len(groups_by_family[family])),
    )


def _eligible_label_holdout_families(
    groups: Sequence[str],
    groups_by_family: dict[str, list[str]],
    *,
    target_count: int,
    target_candidate_count: int,
    decoy_candidate_count: int,
) -> list[str]:
    return sorted(
        [
            family
            for family, family_groups in groups_by_family.items()
            if len(family_groups) >= int(target_count) + int(target_candidate_count)
            and len(groups) - len(family_groups) >= int(decoy_candidate_count) + 1
        ],
        key=lambda family: (family, len(groups_by_family[family])),
    )


def _literal_source_family_assignments(groups: Sequence[str]) -> dict[str, str] | None:
    assignments: dict[str, str] = {}
    for group in groups:
        family = _literal_source_family(str(group))
        if family is None:
            return None
        assignments[str(group)] = family
    if len(set(assignments.values())) < 2:
        return None
    return assignments


def _literal_source_family(group: str) -> str | None:
    if group.startswith("worker") and group[6:].isdigit():
        return None
    for marker in ("_worker", "-worker", "::worker", "/worker"):
        if marker in group:
            prefix = group.split(marker, 1)[0]
            return prefix or None
    for separator in ("::", "|", "/"):
        if separator in group:
            prefix = group.split(separator, 1)[0]
            return prefix or None
    return None


def _cluster_source_family_assignments(
    centroids: np.ndarray,
    groups: Sequence[str],
    *,
    source_family_count: int,
) -> dict[str, str]:
    family_count = min(max(2, int(source_family_count)), len(groups))
    embeddings = normalize_rows(np.asarray(centroids, dtype=float))
    distances = _cosine_distance_matrix(embeddings, embeddings)
    anchor_groups = _target_order_by_isolation(groups, distances)[:family_count]
    centers = np.vstack([embeddings[list(groups).index(group)] for group in anchor_groups])
    labels = np.zeros(len(groups), dtype=int)
    for _iteration in range(12):
        center_distances = _cosine_distance_matrix(centers, embeddings)
        labels = np.argmin(center_distances, axis=0)
        for label in range(family_count):
            if np.any(labels == label):
                continue
            farthest_index = int(np.argmax(np.min(center_distances, axis=0)))
            labels[farthest_index] = label
        next_centers = []
        for label in range(family_count):
            members = embeddings[labels == label]
            next_centers.append(np.mean(members, axis=0) if len(members) else centers[label])
        next_centers = normalize_rows(np.vstack(next_centers))
        if np.allclose(next_centers, centers):
            break
        centers = next_centers
    return {str(group): f"cluster_{int(labels[index]):02d}" for index, group in enumerate(groups)}


def _rotating_take(values: Sequence[str], *, start: int, count: int) -> list[str]:
    if not values:
        return []
    return [values[(start + offset) % len(values)] for offset in range(count)]


def _balanced_counts(total: int, buckets: int) -> list[int]:
    buckets = max(1, int(buckets))
    total = max(1, int(total))
    base = total // buckets
    remainder = total % buckets
    counts = [base + (1 if index < remainder else 0) for index in range(buckets)]
    return [max(1, count) for count in counts]
