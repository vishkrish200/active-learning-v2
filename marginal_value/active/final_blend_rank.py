from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.active.baselines import (
    blended_kcenter_greedy_quality_gated_order,
    old_novelty_only_order,
)
from marginal_value.active.embedding_cache import (
    SUPPORTED_REPRESENTATIONS,
    embedding_cache_dir_from_config,
    load_embedding_lookup,
)
from marginal_value.active.registry import (
    ClipRecord,
    audit_clip_registry_coverage_from_config,
    load_clip_registry_from_config,
)
from marginal_value.indexing.knn_features import normalize_rows
from marginal_value.logging_utils import log_event
from marginal_value.preprocessing.quality import compute_quality_features, load_modal_jsonl_imu
from marginal_value.ranking.baseline_ranker import annotate_cluster_features, compute_batch_clusters, minmax
from marginal_value.submit.finalize_submission import finalize_submission_ids


DEFAULT_TOP_K_VALUES = (10, 50, 100, 200)


def run_active_final_blend_rank(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_active_final_blend_rank_config(config)
    if bool(config["ranking"].get("budgeted_candidate_only", False)):
        return _run_budgeted_candidate_only_final_blend_rank(config, smoke=smoke)

    mode = "smoke" if smoke else "full"
    output_dir = Path(str(config["artifacts"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("active_final_blend_rank", "start", mode=mode)

    registry = load_clip_registry_from_config(config)
    registry_coverage = audit_clip_registry_coverage_from_config(config, registry=registry)
    ranking_config = config["ranking"]
    support_split = str(ranking_config.get("support_split", "pretrain"))
    query_split = str(ranking_config.get("query_split", "new"))
    support_clips = sorted(registry.clips_for_split(support_split), key=lambda clip: clip.sample_id)
    query_clips = sorted(registry.clips_for_split(query_split), key=lambda clip: clip.sample_id)
    if smoke:
        support_clips = support_clips[: int(config["execution"].get("smoke_support_samples", 256))]
        query_clips = query_clips[: int(config["execution"].get("smoke_query_samples", 64))]
    if not support_clips:
        raise ValueError(f"No support clips available for split '{support_split}'.")
    if not query_clips:
        raise ValueError(f"No query clips available for split '{query_split}'.")

    left_rep = str(ranking_config["left_representation"])
    right_rep = str(ranking_config["right_representation"])
    representations = _unique([left_rep, right_rep])
    embedding_result = load_embedding_lookup(
        [*support_clips, *query_clips],
        representations=representations,
        sample_rate=float(ranking_config.get("sample_rate", config.get("quality", {}).get("sample_rate", 30.0))),
        raw_shape_max_samples=ranking_config.get("raw_shape_max_samples"),
        cache_dir=embedding_cache_dir_from_config(config),
        component="active_final_blend_rank",
        mode=mode,
        representation_options=_representation_options(config, ranking_config),
    )
    embeddings = embedding_result.embeddings

    support_ids = [clip.sample_id for clip in support_clips]
    query_ids = [clip.sample_id for clip in query_clips]
    left_support = _stack(embeddings[left_rep], support_ids)
    left_query = _stack(embeddings[left_rep], query_ids)
    right_support = _stack(embeddings[right_rep], support_ids)
    right_query = _stack(embeddings[right_rep], query_ids)

    old_k = min(max(1, int(ranking_config.get("old_novelty_k", 10))), len(support_ids))
    _left_order, left_novelty = old_novelty_only_order(left_support, left_query, k=old_k)
    _right_order, right_novelty = old_novelty_only_order(right_support, right_query, k=old_k)
    rows = _candidate_rows(
        clips=query_clips,
        left_representation=left_rep,
        right_representation=right_rep,
        left_novelty=left_novelty,
        right_novelty=right_novelty,
        quality_config=config.get("quality", {}),
        ranking_config=ranking_config,
    )
    cluster_ids = compute_batch_clusters(
        left_query,
        similarity_threshold=float(ranking_config.get("cluster_similarity_threshold", 0.995)),
    )
    rows = annotate_cluster_features(rows, left_query, cluster_ids)

    alpha = float(ranking_config.get("alpha", 0.5))
    quality_threshold = float(ranking_config.get("quality_threshold", 0.85))
    max_stationary_fraction = _optional_float(ranking_config.get("max_stationary_fraction", 0.90))
    max_abs_value = _optional_float(ranking_config.get("max_abs_value", 60.0))
    order = blended_kcenter_greedy_quality_gated_order(
        left_support,
        left_query,
        right_support,
        right_query,
        rows,
        alpha=alpha,
        quality_threshold=quality_threshold,
        max_stationary_fraction=max_stationary_fraction,
        max_abs_value=max_abs_value,
    )
    ranked_rows = _ranked_rows(rows, order, selector_name=_selector_name(left_rep, right_rep, alpha))

    submission_path = output_dir / f"active_final_blend_submission_{mode}.csv"
    diagnostics_path = output_dir / f"active_final_blend_diagnostics_{mode}.csv"
    report_path = output_dir / f"active_final_blend_report_{mode}.json"
    finalized_worker_id_path = output_dir / f"active_final_blend_submission_{mode}_worker_id.csv"
    finalized_new_worker_id_path = output_dir / f"active_final_blend_submission_{mode}_new_worker_id.csv"

    _write_rows(submission_path, _submission_rows(ranked_rows))
    _write_rows(diagnostics_path, ranked_rows)
    manifest_path = _manifest_path(config, query_split)
    finalize_submission_ids(
        submission_path=submission_path,
        manifest_path=manifest_path,
        output_path=finalized_worker_id_path,
        input_id_column="worker_id",
        output_id_column="worker_id",
    )
    finalize_submission_ids(
        submission_path=submission_path,
        manifest_path=manifest_path,
        output_path=finalized_new_worker_id_path,
        input_id_column="worker_id",
        output_id_column="new_worker_id",
    )

    report = {
        "mode": mode,
        "selector": _selector_name(left_rep, right_rep, alpha),
        "support_split": support_split,
        "query_split": query_split,
        "n_support": len(support_clips),
        "n_query": len(query_clips),
        "representations": representations,
        "left_representation": left_rep,
        "right_representation": right_rep,
        "alpha": alpha,
        "old_novelty_k": old_k,
        "quality_threshold": quality_threshold,
        "max_stationary_fraction": max_stationary_fraction,
        "max_abs_value": max_abs_value,
        "embedding_cache": embedding_result.report(),
        "registry_coverage": _deduplicate_coverage_aliases(registry_coverage),
        "topk_quality": _topk_quality_summary(
            ranked_rows,
            k_values=[int(k) for k in ranking_config.get("top_k_values", DEFAULT_TOP_K_VALUES)],
        ),
        "artifacts": {
            "submission": str(submission_path),
            "diagnostics": str(diagnostics_path),
            "finalized_worker_id": str(finalized_worker_id_path),
            "finalized_new_worker_id": str(finalized_new_worker_id_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "selector": report["selector"],
        "n_support": len(support_clips),
        "n_query": len(query_clips),
        "submission_path": str(submission_path),
        "diagnostics_path": str(diagnostics_path),
        "finalized_worker_id_path": str(finalized_worker_id_path),
        "finalized_new_worker_id_path": str(finalized_new_worker_id_path),
        "report_path": str(report_path),
    }
    log_event("active_final_blend_rank", "done", **result)
    return result


def validate_active_final_blend_rank_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    ranking = _required_mapping(config, "ranking")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Active final blend ranking must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    support_split = str(ranking.get("support_split", "pretrain"))
    query_split = str(ranking.get("query_split", "new"))
    if support_split == query_split:
        raise ValueError("ranking.support_split and ranking.query_split must differ.")
    for split in (support_split, query_split):
        if split not in manifests:
            raise ValueError(f"data.manifests must include split '{split}'.")
    left_rep = str(ranking.get("left_representation", ""))
    right_rep = str(ranking.get("right_representation", ""))
    missing = [name for name, value in (("left_representation", left_rep), ("right_representation", right_rep)) if not value]
    if missing:
        raise ValueError(f"ranking is missing required representation fields: {missing}")
    unsupported = {left_rep, right_rep} - set(SUPPORTED_REPRESENTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported final blend representations: {sorted(unsupported)}")
    alpha = float(ranking.get("alpha", 0.5))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("ranking.alpha must be in [0, 1].")
    if int(ranking.get("old_novelty_k", 10)) <= 0:
        raise ValueError("ranking.old_novelty_k must be positive.")
    if not 0.0 <= float(ranking.get("quality_threshold", 0.85)) <= 1.0:
        raise ValueError("ranking.quality_threshold must be in [0, 1].")
    max_stationary_fraction = ranking.get("max_stationary_fraction", 0.90)
    if max_stationary_fraction is not None and not 0.0 <= float(max_stationary_fraction) <= 1.0:
        raise ValueError("ranking.max_stationary_fraction must be in [0, 1] when provided.")
    max_abs_value = ranking.get("max_abs_value", 60.0)
    if max_abs_value is not None and float(max_abs_value) < 0.0:
        raise ValueError("ranking.max_abs_value must be non-negative when provided.")
    for value in ranking.get("top_k_values", DEFAULT_TOP_K_VALUES):
        if int(value) <= 0:
            raise ValueError("ranking.top_k_values must contain positive integers.")
    if bool(ranking.get("budgeted_candidate_only", False)):
        if not str(ranking.get("left_support_shard_dir", "")).strip():
            raise ValueError("budgeted candidate-only ranking requires ranking.left_support_shard_dir.")
        if ranking.get("max_query_clips") is not None and int(ranking["max_query_clips"]) <= 0:
            raise ValueError("ranking.max_query_clips must be positive when provided.")
        if int(ranking.get("min_left_support_clips", 1)) <= 0:
            raise ValueError("ranking.min_left_support_clips must be positive.")
        if ranking.get("left_support_max_shards") is not None and int(ranking["left_support_max_shards"]) <= 0:
            raise ValueError("ranking.left_support_max_shards must be positive when provided.")
        if ranking.get("right_support_max_clips") is not None and int(ranking["right_support_max_clips"]) <= 0:
            raise ValueError("ranking.right_support_max_clips must be positive when provided.")
        if int(ranking.get("right_embedding_workers", 1)) <= 0:
            raise ValueError("ranking.right_embedding_workers must be positive.")


def _run_budgeted_candidate_only_final_blend_rank(config: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    mode = "smoke" if smoke else "full"
    output_dir = Path(str(config["artifacts"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("active_final_blend_rank", "budgeted_start", mode=mode)

    registry = load_clip_registry_from_config(config)
    registry_coverage = audit_clip_registry_coverage_from_config(config, registry=registry)
    ranking_config = config["ranking"]
    support_split = str(ranking_config.get("support_split", "pretrain"))
    query_split = str(ranking_config.get("query_split", "new"))
    all_support_clips = sorted(registry.clips_for_split(support_split), key=lambda clip: clip.sample_id)
    query_clips = sorted(registry.clips_for_split(query_split), key=lambda clip: clip.sample_id)
    if smoke:
        query_clips = query_clips[: int(config["execution"].get("smoke_query_samples", 64))]
    else:
        query_clips = _apply_full_query_cap(query_clips, ranking_config)
    if not all_support_clips:
        raise ValueError(f"No support clips available for split '{support_split}'.")
    if not query_clips:
        raise ValueError(f"No query clips available for split '{query_split}'.")

    left_rep = str(ranking_config["left_representation"])
    right_rep = str(ranking_config["right_representation"])
    support_ids_allowed = {clip.sample_id for clip in all_support_clips}
    max_left_shards = _optional_int(ranking_config.get("left_support_max_shards"))
    if smoke and ranking_config.get("smoke_left_support_max_shards") is not None:
        max_left_shards = int(ranking_config["smoke_left_support_max_shards"])
    partial_left = _load_partial_embedding_shards(
        Path(str(ranking_config["left_support_shard_dir"])),
        representations=[left_rep],
        allowed_sample_ids=support_ids_allowed,
        max_shards=max_left_shards,
    )
    left_support = partial_left["embeddings"][left_rep]
    left_support_ids = list(partial_left["sample_ids"])
    if smoke:
        smoke_support = int(config["execution"].get("smoke_support_samples", 256))
        left_support = left_support[:smoke_support]
        left_support_ids = left_support_ids[:smoke_support]
    min_left_support = 1 if smoke else int(ranking_config.get("min_left_support_clips", 1))
    if len(left_support) < min_left_support:
        raise ValueError(
            f"Budgeted ranking found only {len(left_support)} cached '{left_rep}' support embeddings; "
            f"required at least {min_left_support}."
        )

    representations = _unique([left_rep, right_rep])
    query_result = load_embedding_lookup(
        query_clips,
        representations=representations,
        sample_rate=float(ranking_config.get("sample_rate", config.get("quality", {}).get("sample_rate", 30.0))),
        raw_shape_max_samples=ranking_config.get("raw_shape_max_samples"),
        cache_dir=_candidate_embedding_cache_dir(config, ranking_config),
        component="active_final_blend_rank",
        mode=f"{mode}_query",
        representation_options=_representation_options(config, ranking_config),
        workers=int(ranking_config.get("query_embedding_workers", 1)),
    )
    query_ids = [clip.sample_id for clip in query_clips]
    left_query = _stack(query_result.embeddings[left_rep], query_ids)
    right_query = _stack(query_result.embeddings[right_rep], query_ids)

    right_support_clips = _right_support_clips_for_budgeted_rank(all_support_clips, config=config, smoke=smoke)
    right_result = load_embedding_lookup(
        right_support_clips,
        representations=[right_rep],
        sample_rate=float(ranking_config.get("sample_rate", config.get("quality", {}).get("sample_rate", 30.0))),
        raw_shape_max_samples=ranking_config.get("raw_shape_max_samples"),
        cache_dir=_right_support_embedding_cache_dir(config, ranking_config),
        component="active_final_blend_rank",
        mode=f"{mode}_right_support",
        representation_options=_representation_options(config, ranking_config),
        workers=int(ranking_config.get("right_embedding_workers", 1)),
    )
    right_support_ids = [clip.sample_id for clip in right_support_clips]
    right_support = _stack(right_result.embeddings[right_rep], right_support_ids)

    old_k_left = min(max(1, int(ranking_config.get("old_novelty_k", 10))), len(left_support_ids))
    old_k_right = min(max(1, int(ranking_config.get("old_novelty_k", 10))), len(right_support_ids))
    _left_order, left_novelty = old_novelty_only_order(left_support, left_query, k=old_k_left)
    _right_order, right_novelty = old_novelty_only_order(right_support, right_query, k=old_k_right)
    rows = _candidate_rows(
        clips=query_clips,
        left_representation=left_rep,
        right_representation=right_rep,
        left_novelty=left_novelty,
        right_novelty=right_novelty,
        quality_config=config.get("quality", {}),
        ranking_config=ranking_config,
    )
    cluster_ids = compute_batch_clusters(
        left_query,
        similarity_threshold=float(ranking_config.get("cluster_similarity_threshold", 0.995)),
    )
    rows = annotate_cluster_features(rows, left_query, cluster_ids)

    alpha = float(ranking_config.get("alpha", 0.5))
    quality_threshold = float(ranking_config.get("quality_threshold", 0.85))
    max_stationary_fraction = _optional_float(ranking_config.get("max_stationary_fraction", 0.90))
    max_abs_value = _optional_float(ranking_config.get("max_abs_value", 60.0))
    order = blended_kcenter_greedy_quality_gated_order(
        left_support,
        left_query,
        right_support,
        right_query,
        rows,
        alpha=alpha,
        quality_threshold=quality_threshold,
        max_stationary_fraction=max_stationary_fraction,
        max_abs_value=max_abs_value,
    )
    ranked_rows = _ranked_rows(rows, order, selector_name=_selector_name(left_rep, right_rep, alpha))

    submission_path = output_dir / f"active_final_blend_submission_{mode}.csv"
    diagnostics_path = output_dir / f"active_final_blend_diagnostics_{mode}.csv"
    report_path = output_dir / f"active_final_blend_report_{mode}.json"
    finalized_worker_id_path = output_dir / f"active_final_blend_submission_{mode}_worker_id.csv"
    finalized_new_worker_id_path = output_dir / f"active_final_blend_submission_{mode}_new_worker_id.csv"

    _write_rows(submission_path, _submission_rows(ranked_rows))
    _write_rows(diagnostics_path, ranked_rows)
    manifest_path = _manifest_path(config, query_split)
    finalize_submission_ids(
        submission_path=submission_path,
        manifest_path=manifest_path,
        output_path=finalized_worker_id_path,
        input_id_column="worker_id",
        output_id_column="worker_id",
    )
    finalize_submission_ids(
        submission_path=submission_path,
        manifest_path=manifest_path,
        output_path=finalized_new_worker_id_path,
        input_id_column="worker_id",
        output_id_column="new_worker_id",
    )

    report = {
        "mode": mode,
        "selector": _selector_name(left_rep, right_rep, alpha),
        "ranking_mode": "budgeted_candidate_only",
        "support_split": support_split,
        "query_split": query_split,
        "n_support": len(right_support_clips),
        "n_left_support": len(left_support),
        "n_right_support": len(right_support),
        "n_query": len(query_clips),
        "representations": representations,
        "left_representation": left_rep,
        "right_representation": right_rep,
        "alpha": alpha,
        "old_novelty_k_left": old_k_left,
        "old_novelty_k_right": old_k_right,
        "quality_threshold": quality_threshold,
        "max_stationary_fraction": max_stationary_fraction,
        "max_abs_value": max_abs_value,
        "left_support_cache": {
            "status": "partial_shard_hit",
            "path": str(ranking_config["left_support_shard_dir"]),
            "n_clips": int(len(left_support)),
        },
        "query_embedding_cache": query_result.report(),
        "right_support_embedding_cache": right_result.report(),
        "right_support_seed": None
        if ranking_config.get("right_support_seed") is None
        else int(ranking_config["right_support_seed"]),
        "registry_coverage": _deduplicate_coverage_aliases(registry_coverage),
        "topk_quality": _topk_quality_summary(
            ranked_rows,
            k_values=[int(k) for k in ranking_config.get("top_k_values", DEFAULT_TOP_K_VALUES)],
        ),
        "artifacts": {
            "submission": str(submission_path),
            "diagnostics": str(diagnostics_path),
            "finalized_worker_id": str(finalized_worker_id_path),
            "finalized_new_worker_id": str(finalized_new_worker_id_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "mode": mode,
        "selector": report["selector"],
        "ranking_mode": "budgeted_candidate_only",
        "n_support": len(right_support_clips),
        "n_left_support": len(left_support),
        "n_query": len(query_clips),
        "submission_path": str(submission_path),
        "diagnostics_path": str(diagnostics_path),
        "finalized_worker_id_path": str(finalized_worker_id_path),
        "finalized_new_worker_id_path": str(finalized_new_worker_id_path),
        "report_path": str(report_path),
    }
    log_event("active_final_blend_rank", "budgeted_done", **result)
    return result


def _candidate_rows(
    *,
    clips: Sequence[ClipRecord],
    left_representation: str,
    right_representation: str,
    left_novelty: np.ndarray,
    right_novelty: np.ndarray,
    quality_config: Mapping[str, Any],
    ranking_config: Mapping[str, Any],
) -> list[dict[str, object]]:
    left_scores = minmax(left_novelty)
    right_scores = minmax(right_novelty)
    alpha = float(ranking_config.get("alpha", 0.5))
    blend_scores = alpha * left_scores + (1.0 - alpha) * right_scores
    rows: list[dict[str, object]] = []
    for idx, clip in enumerate(clips):
        quality = _quality_for_clip(
            clip,
            sample_rate=float(quality_config.get("sample_rate", ranking_config.get("sample_rate", 30.0))),
            max_samples=_optional_int(quality_config.get("max_samples_per_clip")),
        )
        quality_score = _safe_float(quality.get("quality_score", 1.0), default=1.0)
        stationary_fraction = _safe_float(quality.get("stationary_fraction", 0.0))
        max_abs_value = _safe_float(quality.get("max_abs_value", 0.0))
        quality_gate_pass = quality_score >= float(ranking_config.get("quality_threshold", 0.85))
        stationary_pass = (
            ranking_config.get("max_stationary_fraction", 0.90) is None
            or stationary_fraction <= float(ranking_config.get("max_stationary_fraction", 0.90))
        )
        abs_value_pass = (
            ranking_config.get("max_abs_value", 60.0) is None
            or max_abs_value <= float(ranking_config.get("max_abs_value", 60.0))
        )
        failures = []
        if not quality_gate_pass:
            failures.append("quality")
        if not stationary_pass:
            failures.append("stationary")
        if not abs_value_pass:
            failures.append("max_abs_value")
        rows.append(
            {
                "candidate_index": int(idx),
                "worker_id": clip.sample_id,
                "sample_id": clip.sample_id,
                "source_worker_id": clip.worker_id,
                "source_group_id": clip.source_group_id,
                "url": clip.url,
                "raw_path": str(clip.raw_path),
                "quality_score": quality_score,
                "stationary_fraction": stationary_fraction,
                "max_abs_value": max_abs_value,
                "quality_gate_pass": bool(quality_gate_pass),
                "physical_validity_pass": bool(stationary_pass and abs_value_pass),
                "physical_validity_failure_reasons": ",".join(failures),
                f"old_knn_distance_{left_representation}": float(left_novelty[idx]),
                f"old_knn_distance_{right_representation}": float(right_novelty[idx]),
                f"old_novelty_score_{left_representation}": float(left_scores[idx]),
                f"old_novelty_score_{right_representation}": float(right_scores[idx]),
                "blend_old_novelty_score": float(blend_scores[idx]),
                "ranker_score": float(blend_scores[idx]),
                "final_score": float(quality_score * blend_scores[idx]),
            }
        )
    return rows


def _apply_full_query_cap(clips: Sequence[ClipRecord], ranking_config: Mapping[str, Any]) -> list[ClipRecord]:
    max_query_clips = ranking_config.get("max_query_clips")
    if max_query_clips is None:
        return list(clips)
    max_count = int(max_query_clips)
    if len(clips) > max_count and bool(ranking_config.get("fail_if_query_exceeds_max", True)):
        raise ValueError(
            f"Query split has {len(clips)} clips, exceeding ranking.max_query_clips={max_count}."
        )
    return list(clips[:max_count])


def _right_support_clips_for_budgeted_rank(
    clips: Sequence[ClipRecord],
    *,
    config: Mapping[str, Any],
    smoke: bool,
) -> list[ClipRecord]:
    ranking_config = config["ranking"]
    if smoke:
        return list(clips[: int(config["execution"].get("smoke_support_samples", 256))])
    max_support = ranking_config.get("right_support_max_clips")
    if max_support is None:
        return list(clips)
    max_count = int(max_support)
    if len(clips) > max_count and bool(ranking_config.get("fail_if_right_support_exceeds_max", False)):
        raise ValueError(
            f"Support split has {len(clips)} clips, exceeding ranking.right_support_max_clips={max_count}."
        )
    if len(clips) > max_count and ranking_config.get("right_support_seed") is not None:
        rng = np.random.default_rng(int(ranking_config["right_support_seed"]))
        indices = sorted(int(index) for index in rng.choice(len(clips), size=max_count, replace=False))
        return [clips[index] for index in indices]
    return list(clips[:max_count])


def _candidate_embedding_cache_dir(config: Mapping[str, Any], ranking_config: Mapping[str, Any]) -> Path | None:
    value = ranking_config.get("candidate_cache_dir")
    if value is not None and str(value).strip():
        return Path(str(value))
    return embedding_cache_dir_from_config(config)


def _right_support_embedding_cache_dir(config: Mapping[str, Any], ranking_config: Mapping[str, Any]) -> Path | None:
    value = ranking_config.get("right_support_cache_dir")
    if value is not None and str(value).strip():
        return Path(str(value))
    return embedding_cache_dir_from_config(config)


def _load_partial_embedding_shards(
    shard_dir: Path,
    *,
    representations: Sequence[str],
    allowed_sample_ids: set[str],
    max_shards: int | None,
) -> dict[str, object]:
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if max_shards is not None:
        shard_paths = shard_paths[: int(max_shards)]
    if not shard_paths:
        raise FileNotFoundError(f"No partial embedding shards found in {shard_dir}.")

    sample_ids: list[str] = []
    rows: dict[str, list[np.ndarray]] = {str(rep): [] for rep in representations}
    seen: set[str] = set()
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            shard_ids = [str(sample_id) for sample_id in data["sample_ids"].tolist()]
            keep_indices = [
                index
                for index, sample_id in enumerate(shard_ids)
                if sample_id in allowed_sample_ids and sample_id not in seen
            ]
            if not keep_indices:
                continue
            for index in keep_indices:
                sample_id = shard_ids[int(index)]
                seen.add(sample_id)
                sample_ids.append(sample_id)
            keep = np.asarray(keep_indices, dtype=int)
            for representation in rows:
                key = f"rep__{representation}"
                if key not in data:
                    raise KeyError(f"Partial embedding shard {shard_path} is missing '{key}'.")
                rows[representation].append(np.asarray(data[key], dtype="float32")[keep])
    if not sample_ids:
        raise ValueError(f"Partial embedding shards in {shard_dir} did not contain support split sample IDs.")
    matrices = {
        representation: np.vstack(parts).astype("float32")
        for representation, parts in rows.items()
    }
    return {"sample_ids": sample_ids, "embeddings": matrices}


def _quality_for_clip(clip: ClipRecord, *, sample_rate: float, max_samples: int | None) -> Mapping[str, float]:
    if clip.quality and all(key in clip.quality for key in ("quality_score", "stationary_fraction", "max_abs_value")):
        return clip.quality
    samples, timestamps = load_modal_jsonl_imu(clip.raw_path, max_samples=max_samples)
    return compute_quality_features(samples, timestamps=timestamps, sample_rate=sample_rate)


def _ranked_rows(rows: Sequence[Mapping[str, object]], order: Sequence[int], *, selector_name: str) -> list[dict[str, object]]:
    total = max(1, len(order))
    ranked: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        row = dict(rows[int(idx)])
        row["rank"] = int(rank)
        row["selector"] = selector_name
        row["reranker"] = selector_name
        row["rerank_score"] = float((total - rank + 1) / total)
        row["score"] = row["rerank_score"]
        row["reason_code"] = _reason_code(row)
        ranked.append(row)
    return ranked


def _submission_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    previous_score: float | None = None
    for row in rows:
        score = _safe_float(row.get("rerank_score", row.get("score", 0.0)))
        if previous_score is not None and score >= previous_score:
            score = previous_score - 1.0e-12
        previous_score = score
        output.append(
            {
                "worker_id": row["worker_id"],
                "rank": int(row["rank"]),
                "score": float(score),
                "quality_score": _safe_float(row.get("quality_score", 1.0), default=1.0),
                "reason_code": row.get("reason_code", ""),
            }
        )
    return output


def _topk_quality_summary(rows: Sequence[Mapping[str, object]], *, k_values: Sequence[int]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for k in k_values:
        selected = list(rows[: min(int(k), len(rows))])
        n = len(selected)
        clusters = {str(row.get("new_cluster_id", "")) for row in selected}
        quality_fail = sum(not bool(row.get("quality_gate_pass", False)) for row in selected)
        physical_fail = sum(not bool(row.get("physical_validity_pass", False)) for row in selected)
        output[f"k{int(k)}"] = {
            "selected_count": float(n),
            "quality_failure_rate": _fraction(quality_fail, n),
            "physical_failure_rate": _fraction(physical_fail, n),
            "duplicate_rate": 1.0 - _fraction(len(clusters), n),
            "unique_new_clusters": float(len(clusters)),
        }
    return output


def _stack(lookup: Mapping[str, np.ndarray], sample_ids: Sequence[str]) -> np.ndarray:
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing embeddings for {len(missing)} sample IDs: {preview}")
    return np.vstack([lookup[sample_id] for sample_id in sample_ids]).astype("float32")


def _selector_name(left_representation: str, right_representation: str, alpha: float) -> str:
    alpha_int = int(round(float(alpha) * 10))
    return f"blend_kcenter_{left_representation}_{right_representation}_a{alpha_int:02d}"


def _reason_code(row: Mapping[str, object]) -> str:
    reasons = ["BLEND_KCENTER"]
    if bool(row.get("quality_gate_pass", False)) and bool(row.get("physical_validity_pass", False)):
        reasons.append("HYGIENE_PASS")
    else:
        reasons.append("HYGIENE_FALLBACK")
    if _safe_float(row.get("blend_old_novelty_score", 0.0)) >= 0.75:
        reasons.append("HIGH_OLD_NOVELTY")
    if bool(row.get("is_singleton", False)):
        reasons.append("SINGLETON_CLUSTER")
    return "|".join(reasons)


def _manifest_path(config: Mapping[str, Any], split: str) -> Path:
    data = config["data"]
    path = Path(str(data["manifests"][split]))
    if path.is_absolute():
        return path
    return Path(str(data["root"])) / path


def _representation_options(config: Mapping[str, Any], section: Mapping[str, Any]) -> Mapping[str, object]:
    options: dict[str, object] = {}
    embeddings = config.get("embeddings")
    if isinstance(embeddings, Mapping):
        options.update(dict(embeddings))
    explicit = section.get("representation_options")
    if isinstance(explicit, Mapping):
        options.update(dict(explicit))
    return options


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    row_list = [dict(row) for row in rows]
    fieldnames = sorted({key for row in row_list for key in row})
    if "worker_id" in fieldnames:
        fieldnames = ["worker_id", *[field for field in fieldnames if field != "worker_id"]]
    if "rank" in fieldnames:
        fieldnames = [field for field in fieldnames if field != "rank"]
        insert_at = 1 if fieldnames and fieldnames[0] == "worker_id" else 0
        fieldnames.insert(insert_at, "rank")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _deduplicate_coverage_aliases(coverage: Mapping[str, Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    seen_ids: set[int] = set()
    for key in ("old", "new"):
        if key in coverage:
            output[key] = coverage[key]
            seen_ids.add(id(coverage[key]))
    for key, value in coverage.items():
        if key not in output and id(value) not in seen_ids:
            output[key] = value
    return output


def _unique(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"active final blend ranking config requires object field '{key}'.")
    return value


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
