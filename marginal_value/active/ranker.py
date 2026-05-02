from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from marginal_value.active.baselines import (
    BASELINE_POLICY_NAMES,
    learned_score_quality_gated_kcenter_order,
)
from marginal_value.active.embedding_cache import SUPPORTED_REPRESENTATIONS, embedding_cache_dir_from_config, load_embedding_lookup
from marginal_value.active.evaluate_active_loop import (
    DEFAULT_REPRESENTATIONS,
    ORACLE_POLICY_NAME,
    _add_oracle_fractions,
    _balanced_metrics,
    _candidate_clusters,
    _coverage_by_representation,
    _deduplicate_coverage_aliases,
    _policy_orders_for_episode,
    _policy_summary,
    _registry_coverage_summary,
    _selection_audit_row,
    _stack,
    load_active_episodes,
)
from marginal_value.active.label_gain import compute_gated_labels
from marginal_value.active.registry import (
    ClipRegistry,
    audit_clip_registry_coverage_from_config,
    load_clip_registry_from_config,
)
from marginal_value.indexing.cosine_search import cosine_knn
from marginal_value.indexing.knn_features import normalize_rows
from marginal_value.logging_utils import log_event, log_progress


MODEL_POLICY_NAME = "learned_ridge_balanced_gain"
HYBRID_POLICY_NAME = "learned_ridge_quality_gated_kcenter"

FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "candidate_role",
        "heldout_novel",
        "known_like",
        "near_duplicate",
        "low_quality",
        "hidden_target_membership",
        "target_source_group_id",
        "candidate_target_distance",
        "support_target_distance",
        "coverage_before",
        "coverage_after",
        "gain_if_added_alone",
        "balanced_gain",
        "balanced_relative_gain",
        "gain_after_greedy_prefix",
        "oracle_prefix_rank",
        "oracle_selected",
        "url",
        "raw_path",
        "split",
    }
)


@dataclass(frozen=True)
class ActiveRankerFeatureTable:
    rows: list[dict[str, object]]
    values: np.ndarray
    labels: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class RidgeRankerModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float


def run_active_ranker_train_eval(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_active_ranker_config(config)
    mode = "smoke" if smoke else "full"
    output_dir = Path(str(config["artifacts"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("active_ranker", "start", mode=mode)

    registry = load_clip_registry_from_config(config)
    registry_coverage = audit_clip_registry_coverage_from_config(config, registry=registry)
    episodes = load_active_episodes(_episode_path(config))
    if smoke:
        episodes = episodes[: int(config["execution"].get("smoke_max_episodes", 1))]
    if not episodes:
        raise ValueError("Active ranker training requires at least one episode.")

    ranker_config = config["ranker"]
    representations = [str(rep) for rep in ranker_config.get("representations", DEFAULT_REPRESENTATIONS)]
    primary_representation = str(ranker_config.get("primary_representation", representations[0]))
    if primary_representation not in representations:
        raise ValueError("ranker.primary_representation must be included in ranker.representations.")
    used_ids = _episode_sample_ids(episodes)
    embedding_result = load_embedding_lookup(
        [registry.by_sample_id[sample_id] for sample_id in sorted(used_ids)],
        representations=representations,
        sample_rate=float(ranker_config.get("sample_rate", 30.0)),
        raw_shape_max_samples=ranker_config.get("raw_shape_max_samples"),
        cache_dir=embedding_cache_dir_from_config(config),
        component="active_ranker",
        mode=mode,
        representation_options=_representation_options(config, ranker_config),
    )
    label_rows = _read_label_rows(_labels_path(config))
    label_column = _ranker_label_column(ranker_config)
    hygiene_gate_config = ranker_config.get("hygiene_gate", {})
    if isinstance(hygiene_gate_config, Mapping) and (hygiene_gate_config or _label_column_missing(label_rows, label_column)):
        label_rows = compute_gated_labels(label_rows, component="active_ranker", **dict(hygiene_gate_config))
    table = build_active_ranker_feature_table(
        episodes=episodes,
        registry=registry,
        embeddings=embedding_result.embeddings,
        label_rows=label_rows,
        representations=representations,
        label_column=label_column,
        old_novelty_k=int(ranker_config.get("old_novelty_k", 10)),
        cluster_similarity_threshold=float(ranker_config.get("cluster_similarity_threshold", 0.995)),
    )
    split = episode_level_split(
        [episode.episode_id for episode in episodes],
        train_count=_optional_int(ranker_config.get("train_episode_count")),
        validation_count=_optional_int(ranker_config.get("validation_episode_count")),
        test_count=_optional_int(ranker_config.get("test_episode_count")),
    )
    train_mask = np.asarray([row["episode_id"] in set(split["train"]) for row in table.rows], dtype=bool)
    if not np.any(train_mask):
        raise ValueError("Active ranker split produced no training rows.")
    model = fit_ridge_regressor(table.values[train_mask], table.labels[train_mask], l2=float(ranker_config.get("ridge_l2", 1.0)))
    scores = score_ridge_regressor(model, table.values)

    model_path = output_dir / f"active_ranker_model_{mode}.json"
    scores_path = output_dir / f"active_ranker_scores_{mode}.csv"
    coverage_path = output_dir / f"active_ranker_coverage_by_episode_{mode}.csv"
    selection_path = output_dir / f"active_ranker_topk_selection_audit_{mode}.csv"
    report_path = output_dir / f"active_ranker_report_{mode}.json"

    score_rows = _score_rows(table, scores, split)
    _write_csv(scores_path, score_rows)
    write_ridge_ranker_model(
        model_path,
        model,
        feature_names=table.feature_names,
        metadata={
            "target": str(ranker_config.get("target", "balanced_gain")),
            "label_column": label_column,
            "mode": mode,
            "split": split,
            "model_policy_name": MODEL_POLICY_NAME,
        },
    )

    coverage_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    split_reports: dict[str, dict[str, object]] = {}
    score_by_episode_candidate = {
        (str(row["episode_id"]), int(row["candidate_index"])): float(score)
        for row, score in zip(table.rows, scores)
    }
    baseline_policies = [
        str(policy)
        for policy in ranker_config.get(
            "baseline_policies",
            [*BASELINE_POLICY_NAMES, ORACLE_POLICY_NAME],
        )
    ]
    k_values = [int(k) for k in ranker_config.get("k_values", [5, 10, 25, 50, 100])]
    for split_name in ("train", "validation", "test"):
        split_episode_ids = set(split[split_name])
        split_episodes = [episode for episode in episodes if episode.episode_id in split_episode_ids]
        split_coverage_rows, split_selection_rows = _evaluate_ranker_split(
            episodes=split_episodes,
            registry=registry,
            embeddings=embedding_result.embeddings,
            representations=representations,
            primary_representation=primary_representation,
            score_by_episode_candidate=score_by_episode_candidate,
            baseline_policies=baseline_policies,
            k_values=k_values,
            quality_threshold=float(ranker_config.get("quality_threshold", 0.45)),
            old_novelty_k=int(ranker_config.get("old_novelty_k", 10)),
            cluster_similarity_threshold=float(ranker_config.get("cluster_similarity_threshold", 0.995)),
            random_seed=int(ranker_config.get("random_seed", 20260429)),
            hybrid_config=ranker_config.get("hybrid_policy", {}),
            split_name=split_name,
        )
        _add_oracle_fractions(split_coverage_rows)
        split_reports[split_name] = {
            "n_episodes": len(split_episodes),
            "policies": _policy_summary(split_coverage_rows),
        }
        coverage_rows.extend(split_coverage_rows)
        selection_rows.extend(split_selection_rows)
    _write_csv(coverage_path, coverage_rows)
    _write_csv(selection_path, selection_rows)

    report = {
        "mode": mode,
        "n_episodes": len(episodes),
        "n_rows": len(table.rows),
        "representations": representations,
        "primary_representation": primary_representation,
        "model_policy_name": MODEL_POLICY_NAME,
        "hybrid_policy_name": HYBRID_POLICY_NAME,
        "hybrid_policy": dict(ranker_config.get("hybrid_policy", {})),
        "label_column": label_column,
        "split": split,
        "embedding_cache": embedding_result.report(),
        "registry_coverage": _deduplicate_coverage_aliases(registry_coverage),
        "registry_coverage_summary": _registry_coverage_summary(registry_coverage),
        "feature_names": table.feature_names,
        "feature_summary": _feature_summary(table),
        "target_summary": _target_summary(table.labels, table.rows),
        "feature_leakage_check": _feature_leakage_check(table.feature_names),
        "splits": split_reports,
        "artifacts": {
            "model": str(model_path),
            "scores": str(scores_path),
            "coverage_by_episode": str(coverage_path),
            "topk_selection_audit": str(selection_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "n_episodes": len(episodes),
        "n_rows": len(table.rows),
        "report_path": str(report_path),
        "model_path": str(model_path),
        "scores_csv_path": str(scores_path),
        "coverage_by_episode_path": str(coverage_path),
        "selection_audit_path": str(selection_path),
    }
    log_event("active_ranker", "done", **result)
    return result


def build_active_ranker_feature_table(
    *,
    episodes: Sequence[Any],
    registry: ClipRegistry,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    label_rows: Sequence[Mapping[str, object]],
    representations: Sequence[str],
    target: str | None = None,
    label_column: str | None = None,
    old_novelty_k: int = 10,
    cluster_similarity_threshold: float = 0.995,
) -> ActiveRankerFeatureTable:
    selected_label_column = str(label_column or target or "gated_balanced_gain")
    label_by_key = {
        (str(row["episode_id"]), str(row["sample_id"])): row
        for row in label_rows
    }
    rows: list[dict[str, object]] = []
    feature_dicts: list[dict[str, float]] = []
    labels: list[float] = []
    progress_every = max(1, len(episodes) // 10)
    for episode_index, episode in enumerate(episodes, start=1):
        candidate_ids = list(episode.candidate_clip_ids)
        per_rep = _per_representation_episode_features(
            episode=episode,
            embeddings=embeddings,
            representations=representations,
            old_novelty_k=old_novelty_k,
        )
        primary_vectors = _stack(embeddings[str(representations[0])], candidate_ids)
        cluster_ids = _candidate_clusters(primary_vectors, similarity_threshold=cluster_similarity_threshold)
        cluster_counts = Counter(int(cluster_id) for cluster_id in cluster_ids)
        cluster_medoid_distances = _cluster_medoid_distances(primary_vectors, cluster_ids)
        for candidate_index, sample_id in enumerate(candidate_ids):
            label = label_by_key.get((episode.episode_id, sample_id))
            if label is None:
                raise KeyError(f"Missing active-ranker label for {episode.episode_id}/{sample_id}")
            if selected_label_column not in label:
                raise ValueError(
                    f"label_column '{selected_label_column}' not found in label_df. "
                    f"Available columns: {list(label.keys())}. "
                    "Did you run compute_gated_labels() before training?"
                )
            clip = registry.by_sample_id[sample_id]
            feature = _candidate_feature_dict(
                candidate_index=candidate_index,
                clip_quality=clip.quality,
                per_rep=per_rep,
                representations=representations,
                cluster_id=int(cluster_ids[candidate_index]),
                cluster_count=int(cluster_counts[int(cluster_ids[candidate_index])]),
                cluster_medoid_distance=float(cluster_medoid_distances[candidate_index]),
                n_candidates=len(candidate_ids),
            )
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "candidate_index": int(candidate_index),
                    "sample_id": sample_id,
                    "source_group_id": clip.source_group_id,
                    "worker_id": clip.worker_id,
                    "target": float(label[selected_label_column]),
                }
            )
            feature_dicts.append(feature)
            labels.append(float(label[selected_label_column]))
        log_progress(
            "active_ranker",
            "feature_episode_progress",
            index=episode_index,
            total=len(episodes),
            every=progress_every,
            rows=len(rows),
        )
    feature_names = sorted(feature_dicts[0]) if feature_dicts else []
    leakage = _feature_leakage_check(feature_names)
    if int(leakage["forbidden_feature_count"]) > 0:
        raise ValueError(f"Active ranker feature table contains forbidden features: {leakage['forbidden_features']}")
    values = np.asarray([[feature[name] for name in feature_names] for feature in feature_dicts], dtype=np.float32)
    return ActiveRankerFeatureTable(rows=rows, values=values, labels=np.asarray(labels, dtype=np.float32), feature_names=feature_names)


def episode_level_split(
    episode_ids: Sequence[str],
    *,
    train_count: int | None = None,
    validation_count: int | None = None,
    test_count: int | None = None,
) -> dict[str, list[str]]:
    ordered = list(dict.fromkeys(str(episode_id) for episode_id in episode_ids))
    n_episodes = len(ordered)
    if n_episodes == 0:
        return {"train": [], "validation": [], "test": []}
    requested = [train_count, validation_count, test_count]
    if all(value is not None for value in requested) and sum(int(value) for value in requested if value is not None) <= n_episodes:
        train_n = int(train_count or 0)
        validation_n = int(validation_count or 0)
        test_n = int(test_count or 0)
    elif n_episodes == 1:
        train_n, validation_n, test_n = 1, 0, 0
    elif n_episodes == 2:
        train_n, validation_n, test_n = 1, 0, 1
    else:
        train_n = max(1, int(np.floor(0.75 * n_episodes)))
        validation_n = max(0, int(np.floor(0.125 * n_episodes)))
        test_n = max(1, n_episodes - train_n - validation_n)
        if train_n + validation_n + test_n > n_episodes:
            train_n = n_episodes - validation_n - test_n
    train = ordered[:train_n]
    validation = ordered[train_n : train_n + validation_n]
    test = ordered[train_n + validation_n : train_n + validation_n + test_n]
    return {"train": train, "validation": validation, "test": test}


def fit_ridge_regressor(values: np.ndarray, labels: np.ndarray, *, l2: float = 1.0) -> RidgeRankerModel:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be a 2D matrix.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("values and labels must have the same number of rows.")
    if x.shape[0] == 0:
        raise ValueError("fit_ridge_regressor requires at least one row.")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1.0e-12, 1.0, scale)
    z = (x - mean) / scale
    y_mean = float(np.mean(y))
    centered_y = y - y_mean
    penalty = max(0.0, float(l2))
    system = z.T @ z + penalty * np.eye(z.shape[1], dtype=np.float64)
    rhs = z.T @ centered_y
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ rhs
    return RidgeRankerModel(feature_mean=mean, feature_scale=scale, weights=weights, bias=y_mean)


def score_ridge_regressor(model: RidgeRankerModel, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be a 2D matrix.")
    if x.shape[1] != len(model.weights):
        raise ValueError(f"Expected {len(model.weights)} features, got {x.shape[1]}.")
    return ((x - model.feature_mean) / model.feature_scale) @ model.weights + model.bias


def write_ridge_ranker_model(
    path: str | Path,
    model: RidgeRankerModel,
    *,
    feature_names: Sequence[str],
    metadata: Mapping[str, object] | None = None,
) -> None:
    payload = {
        "model_type": "active_ridge_regressor",
        "feature_names": list(feature_names),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "weights": model.weights.tolist(),
        "bias": float(model.bias),
        "metadata": dict(metadata or {}),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_active_ranker_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    episodes = _required_mapping(config, "episodes")
    labels = _required_mapping(config, "labels")
    ranker = _required_mapping(config, "ranker")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Active ranker training must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    if "path" not in episodes:
        raise ValueError("episodes.path is required.")
    if "path" not in labels:
        raise ValueError("labels.path is required.")
    if not manifests:
        raise ValueError("data.manifests must not be empty.")
    for manifest in manifests.values():
        if not str(manifest).startswith("cache/manifests/"):
            raise ValueError("data.manifests paths must be under cache/manifests/.")
    representations = ranker.get("representations", DEFAULT_REPRESENTATIONS)
    if not isinstance(representations, list | tuple) or not representations:
        raise ValueError("ranker.representations must be a non-empty list.")
    unsupported = set(str(rep) for rep in representations) - set(SUPPORTED_REPRESENTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported active-ranker representations: {sorted(unsupported)}")
    label_column = _ranker_label_column(ranker)
    supported_labels = {
        "balanced_gain",
        "balanced_relative_gain",
        "balanced_gain_after_greedy_prefix",
        "gated_balanced_gain",
        "gated_balanced_relative_gain",
        "gated_balanced_gain_after_greedy_prefix",
    }
    if label_column not in supported_labels:
        raise ValueError("ranker.label_column must be a supported marginal-gain label.")
    hygiene_gate = ranker.get("hygiene_gate", {})
    if hygiene_gate:
        if not isinstance(hygiene_gate, Mapping):
            raise ValueError("ranker.hygiene_gate must be an object when provided.")
        for key in ("quality_threshold", "max_stationary_fraction", "max_abs_value", "duplicate_cosine_threshold"):
            if key in hygiene_gate and hygiene_gate[key] is not None and float(hygiene_gate[key]) < 0.0:
                raise ValueError(f"ranker.hygiene_gate.{key} must be non-negative.")
    if float(ranker.get("ridge_l2", 1.0)) < 0.0:
        raise ValueError("ranker.ridge_l2 must be non-negative.")
    for k in ranker.get("k_values", [5, 10, 25, 50, 100]):
        if int(k) <= 0:
            raise ValueError("ranker.k_values must be positive.")
    hybrid = ranker.get("hybrid_policy", {})
    if hybrid:
        if not isinstance(hybrid, Mapping):
            raise ValueError("ranker.hybrid_policy must be an object when provided.")
        if float(hybrid.get("quality_threshold", 0.85)) < 0.0:
            raise ValueError("ranker.hybrid_policy.quality_threshold must be non-negative.")
        if float(hybrid.get("pool_multiplier", 1.5)) < 1.0:
            raise ValueError("ranker.hybrid_policy.pool_multiplier must be at least 1.0.")


def _per_representation_episode_features(
    *,
    episode: Any,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    representations: Sequence[str],
    old_novelty_k: int,
) -> dict[str, dict[str, np.ndarray]]:
    candidate_ids = list(episode.candidate_clip_ids)
    per_rep: dict[str, dict[str, np.ndarray]] = {}
    for representation in representations:
        lookup = embeddings[str(representation)]
        support = _stack(lookup, episode.support_clip_ids)
        candidates = _stack(lookup, candidate_ids)
        distances, _indices = cosine_knn(support, candidates, k=min(max(1, int(old_novelty_k)), len(episode.support_clip_ids)), backend="auto")
        old_novelty = np.mean(distances, axis=1)
        candidate_distances = _candidate_distance_features(candidates)
        rep_features = {
            "old_novelty": old_novelty,
            "old_novelty_score": _minmax(old_novelty),
            "old_min_distance": np.min(distances, axis=1),
            "candidate_nn_distance": candidate_distances["nn_distance"],
            "candidate_mean_distance": candidate_distances["mean_distance"],
        }
        if str(representation) == "ts2vec":
            for k_value in (5, 10):
                k_distances, _k_indices = cosine_knn(
                    support,
                    candidates,
                    k=min(max(1, int(k_value)), len(episode.support_clip_ids)),
                    backend="auto",
                )
                k_novelty = np.mean(k_distances, axis=1)
                rep_features[f"old_novelty_k{k_value}"] = k_novelty
                rep_features[f"old_novelty_score_k{k_value}"] = _minmax(k_novelty)
        per_rep[str(representation)] = rep_features
    return per_rep


def _candidate_feature_dict(
    *,
    candidate_index: int,
    clip_quality: Mapping[str, object],
    per_rep: Mapping[str, Mapping[str, np.ndarray]],
    representations: Sequence[str],
    cluster_id: int,
    cluster_count: int,
    cluster_medoid_distance: float,
    n_candidates: int,
) -> dict[str, float]:
    quality_score = _safe_float(clip_quality.get("quality_score", 1.0), 1.0)
    stationary_fraction = _safe_float(clip_quality.get("stationary_fraction", 0.0), 0.0)
    max_abs_value = _safe_float(clip_quality.get("max_abs_value", 0.0), 0.0)
    feature: dict[str, float] = {
        "quality_score": quality_score,
        "quality_penalty": 1.0 - quality_score,
        "stationary_fraction": stationary_fraction,
        "motion_fraction": 1.0 - stationary_fraction,
        "max_abs_value_scaled": max_abs_value / 60.0,
        "max_abs_over_60": max(0.0, (max_abs_value - 60.0) / 60.0),
        "new_cluster_size": float(cluster_count),
        "new_cluster_fraction": float(cluster_count / max(1, n_candidates)),
        "inverse_new_cluster_size": float(1.0 / max(1, cluster_count)),
        "distance_to_new_cluster_medoid": float(cluster_medoid_distance),
        "duplicate_score": float(max(0.0, 1.0 - cluster_medoid_distance)),
    }
    novelty_values: list[float] = []
    novelty_scores: list[float] = []
    nn_distances: list[float] = []
    for representation in representations:
        rep = str(representation)
        old_novelty = float(per_rep[rep]["old_novelty"][candidate_index])
        old_score = float(per_rep[rep]["old_novelty_score"][candidate_index])
        old_min = float(per_rep[rep]["old_min_distance"][candidate_index])
        nn_distance = float(per_rep[rep]["candidate_nn_distance"][candidate_index])
        mean_distance = float(per_rep[rep]["candidate_mean_distance"][candidate_index])
        feature[f"old_novelty_{rep}"] = old_novelty
        feature[f"old_novelty_score_{rep}"] = old_score
        feature[f"old_min_distance_{rep}"] = old_min
        feature[f"candidate_nn_distance_{rep}"] = nn_distance
        feature[f"candidate_mean_distance_{rep}"] = mean_distance
        feature[f"quality_x_old_novelty_score_{rep}"] = quality_score * old_score
        feature[f"quality_x_candidate_nn_distance_{rep}"] = quality_score * nn_distance
        if rep == "ts2vec":
            for k_value in (5, 10):
                novelty_key = f"old_novelty_k{k_value}"
                score_key = f"old_novelty_score_k{k_value}"
                if novelty_key in per_rep[rep]:
                    feature[f"old_novelty_ts2vec_k{k_value}"] = float(per_rep[rep][novelty_key][candidate_index])
                if score_key in per_rep[rep]:
                    feature[f"old_novelty_score_ts2vec_k{k_value}"] = float(per_rep[rep][score_key][candidate_index])
        novelty_values.append(old_novelty)
        novelty_scores.append(old_score)
        nn_distances.append(nn_distance)
    feature["old_novelty_mean"] = float(np.mean(novelty_values))
    feature["old_novelty_std"] = float(np.std(novelty_values))
    feature["old_novelty_max"] = float(np.max(novelty_values))
    feature["old_novelty_min"] = float(np.min(novelty_values))
    feature["old_novelty_score_mean"] = float(np.mean(novelty_scores))
    feature["old_novelty_score_std"] = float(np.std(novelty_scores))
    feature["candidate_nn_distance_mean"] = float(np.mean(nn_distances))
    feature["candidate_nn_distance_std"] = float(np.std(nn_distances))
    feature["quality_x_old_novelty_score_mean"] = quality_score * feature["old_novelty_score_mean"]
    if "ts2vec" in per_rep and "raw_shape_stats" in per_rep:
        feature["ts2vec_x_raw_shape_old_novelty_score"] = float(
            per_rep["ts2vec"]["old_novelty_score"][candidate_index]
            * per_rep["raw_shape_stats"]["old_novelty_score"][candidate_index]
        )
    return feature


def _evaluate_ranker_split(
    *,
    episodes: Sequence[Any],
    registry: ClipRegistry,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    representations: Sequence[str],
    primary_representation: str,
    score_by_episode_candidate: Mapping[tuple[str, int], float],
    baseline_policies: Sequence[str],
    k_values: Sequence[int],
    quality_threshold: float,
    old_novelty_k: int,
    cluster_similarity_threshold: float,
    random_seed: int,
    hybrid_config: Mapping[str, object],
    split_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    coverage_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    progress_every = max(1, len(episodes) // 10) if episodes else 1
    for episode_index, episode in enumerate(episodes, start=1):
        baseline_orders, candidate_rows = _policy_orders_for_episode(
            episode=episode,
            registry=registry,
            embeddings=embeddings,
            policies=baseline_policies,
            primary_representation=primary_representation,
            quality_threshold=quality_threshold,
            random_seed=random_seed + episode_index,
            old_novelty_k=old_novelty_k,
            cluster_similarity_threshold=cluster_similarity_threshold,
            max_selected=max(k_values) if k_values else len(episode.candidate_clip_ids),
        )
        learned_order = sorted(
            range(len(episode.candidate_clip_ids)),
            key=lambda idx: (
                -float(score_by_episode_candidate.get((episode.episode_id, int(idx)), -np.inf)),
                str(episode.candidate_clip_ids[int(idx)]),
            ),
        )
        model_orders = {MODEL_POLICY_NAME: learned_order}
        if bool(hybrid_config.get("enabled", False)):
            candidate_ids = list(episode.candidate_clip_ids)
            support_embeddings = _stack(embeddings[primary_representation], episode.support_clip_ids)
            candidate_embeddings = _stack(embeddings[primary_representation], candidate_ids)
            hybrid_rows = [
                dict(
                    row,
                    learned_score=float(score_by_episode_candidate.get((episode.episode_id, int(idx)), -np.inf)),
                )
                for idx, row in enumerate(candidate_rows)
            ]
            model_orders[HYBRID_POLICY_NAME] = learned_score_quality_gated_kcenter_order(
                support_embeddings,
                candidate_embeddings,
                hybrid_rows,
                quality_threshold=float(hybrid_config.get("quality_threshold", quality_threshold)),
                max_stationary_fraction=_optional_float(hybrid_config.get("max_stationary_fraction", 0.90)),
                max_abs_value=_optional_float(hybrid_config.get("max_abs_value", 60.0)),
                max_selected=max(k_values) if k_values else len(candidate_ids),
                pool_multiplier=float(hybrid_config.get("pool_multiplier", 1.5)),
            )
        policy_orders = {**model_orders, **baseline_orders}
        for policy_name, order in policy_orders.items():
            for k in k_values:
                selected = list(order[: min(int(k), len(order))])
                rep_metrics = _coverage_by_representation(
                    episode=episode,
                    embeddings=embeddings,
                    representations=representations,
                    selected_indices=selected,
                )
                balanced = _balanced_metrics(rep_metrics.values())
                for representation, metrics in [*rep_metrics.items(), ("balanced", balanced)]:
                    coverage_rows.append(
                        {
                            "split": split_name,
                            "episode_id": episode.episode_id,
                            "policy": policy_name,
                            "k": int(k),
                            "representation": representation,
                            **metrics,
                        }
                    )
                audit = _selection_audit_row(
                    episode=episode,
                    registry=registry,
                    candidate_rows=candidate_rows,
                    selected_indices=selected,
                    policy_name=policy_name,
                    k=int(k),
                    quality_threshold=quality_threshold,
                )
                audit["split"] = split_name
                selection_rows.append(audit)
        log_progress(
            "active_ranker",
            "eval_episode_progress",
            index=episode_index,
            total=len(episodes),
            every=progress_every,
            split=split_name,
        )
    return coverage_rows, selection_rows


def _candidate_distance_features(candidate_embeddings: np.ndarray) -> dict[str, np.ndarray]:
    vectors = normalize_rows(np.asarray(candidate_embeddings, dtype=float))
    n_rows = len(vectors)
    if n_rows <= 1:
        return {"nn_distance": np.ones(n_rows), "mean_distance": np.ones(n_rows)}
    distances = 1.0 - vectors @ vectors.T
    np.fill_diagonal(distances, np.nan)
    return {
        "nn_distance": np.nanmin(distances, axis=1),
        "mean_distance": np.nanmean(distances, axis=1),
    }


def _cluster_medoid_distances(candidate_embeddings: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    vectors = normalize_rows(np.asarray(candidate_embeddings, dtype=float))
    distances = np.zeros(len(vectors), dtype=float)
    for cluster_id in sorted(set(int(value) for value in cluster_ids.tolist())):
        indices = np.flatnonzero(cluster_ids == cluster_id)
        cluster_vectors = vectors[indices]
        if len(indices) <= 1:
            distances[indices] = 1.0
            continue
        centroid = normalize_rows(np.mean(cluster_vectors, axis=0, keepdims=True))[0]
        distances[indices] = 1.0 - cluster_vectors @ centroid
    return distances


def _score_rows(table: ActiveRankerFeatureTable, scores: np.ndarray, split: Mapping[str, Sequence[str]]) -> list[dict[str, object]]:
    split_by_episode = {
        str(episode_id): split_name
        for split_name, episode_ids in split.items()
        for episode_id in episode_ids
    }
    output = []
    for row, score, label in zip(table.rows, scores, table.labels):
        output.append(
            {
                **row,
                "split": split_by_episode.get(str(row["episode_id"]), "unassigned"),
                "learned_score": float(score),
                "target": float(label),
            }
        )
    return output


def _feature_leakage_check(feature_names: Sequence[str]) -> dict[str, object]:
    forbidden = sorted(name for name in feature_names if name in FORBIDDEN_FEATURE_NAMES)
    return {
        "forbidden_feature_count": int(len(forbidden)),
        "forbidden_features": forbidden,
    }


def _feature_summary(table: ActiveRankerFeatureTable) -> dict[str, object]:
    return {
        "n_features": int(len(table.feature_names)),
        "n_rows": int(table.values.shape[0]),
        "feature_mean_abs_max": float(np.max(np.abs(np.mean(table.values, axis=0)))) if table.values.size else 0.0,
    }


def _target_summary(labels: np.ndarray, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    values = np.asarray(labels, dtype=float)
    by_episode: defaultdict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values):
        by_episode[str(row["episode_id"])].append(float(value))
    return {
        "count": int(len(values)),
        "positive_count": int(np.sum(values > 1.0e-12)),
        "positive_rate": float(np.mean(values > 1.0e-12)) if len(values) else 0.0,
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "max": float(np.max(values)) if len(values) else 0.0,
        "episode_positive_rate_mean": float(np.mean([np.mean(np.asarray(vals) > 1.0e-12) for vals in by_episode.values()])) if by_episode else 0.0,
    }


def _read_label_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


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


def _episode_path(config: Mapping[str, Any]) -> Path:
    path = Path(str(config["episodes"]["path"]))
    if path.is_absolute():
        return path
    return Path(str(config["data"]["root"])) / path


def _labels_path(config: Mapping[str, Any]) -> Path:
    path = Path(str(config["labels"]["path"]))
    if path.is_absolute():
        return path
    return Path(str(config["data"]["root"])) / path


def _ranker_label_column(ranker_config: Mapping[str, object]) -> str:
    if "label_column" in ranker_config:
        return str(ranker_config["label_column"])
    if "target" in ranker_config:
        return str(ranker_config["target"])
    return "gated_balanced_gain"


def _label_column_missing(label_rows: Sequence[Mapping[str, object]], label_column: str) -> bool:
    return bool(label_rows) and str(label_column) not in label_rows[0]


def _episode_sample_ids(episodes: Sequence[Any]) -> set[str]:
    sample_ids: set[str] = set()
    for episode in episodes:
        sample_ids.update(episode.support_clip_ids)
        sample_ids.update(episode.candidate_clip_ids)
        sample_ids.update(episode.hidden_target_clip_ids)
    return sample_ids


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _minmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.asarray([], dtype=float)
    lo = float(np.min(array))
    hi = float(np.max(array))
    if hi - lo < 1.0e-12:
        return np.zeros_like(array)
    return (array - lo) / (hi - lo)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"active ranker config requires object field '{key}'.")
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
