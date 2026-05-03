from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from marginal_value.active.baselines import blended_kcenter_greedy_quality_gated_order, old_novelty_only_order
from marginal_value.active.embedding_cache import SUPPORTED_REPRESENTATIONS
from marginal_value.active.final_blend_rank import (
    DEFAULT_TOP_K_VALUES,
    _candidate_rows,
    _deduplicate_coverage_aliases,
    _load_partial_embedding_shards,
    _manifest_path,
    _optional_float,
    _optional_int,
    _ranked_rows,
    _safe_float,
    _selector_name,
    _stack,
    _submission_rows,
    _topk_quality_summary,
    _write_rows,
)
from marginal_value.active.registry import audit_clip_registry_coverage_from_config, load_clip_registry_from_config
from marginal_value.data.shard_reader import iter_shard_arrays, load_shard_manifest
from marginal_value.logging_utils import log_event
from marginal_value.ranking.baseline_ranker import annotate_cluster_features, compute_batch_clusters
from marginal_value.submit.finalize_submission import finalize_submission_ids


def run_active_exact_window_blend_rank(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    validate_active_exact_window_blend_rank_config(config)
    mode = "smoke" if smoke else "full"
    output_dir = Path(str(config["artifacts"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event("active_exact_window_blend_rank", "start", mode=mode)

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
    log_event(
        "active_exact_window_blend_rank",
        "registry_ready",
        mode=mode,
        support_split=support_split,
        query_split=query_split,
        n_support_available=len(all_support_clips),
        n_query=len(query_clips),
    )

    left_rep = str(ranking_config["left_representation"])
    right_rep = str(ranking_config.get("right_representation", "window_mean_std_pool"))
    selector_mode = str(ranking_config.get("selector_mode", "blend_kcenter"))
    if selector_mode not in {"blend_kcenter", "old_novelty_only"}:
        raise ValueError("ranking.selector_mode must be 'blend_kcenter' or 'old_novelty_only'.")
    if right_rep != "window_mean_std_pool":
        raise ValueError("Exact-window blend currently requires right_representation='window_mean_std_pool'.")
    uses_left_cache = selector_mode != "old_novelty_only"

    support_ids_allowed = {clip.sample_id for clip in all_support_clips}
    if uses_left_cache:
        log_event(
            "active_exact_window_blend_rank",
            "left_support_load_start",
            mode=mode,
            representation=left_rep,
            shard_dir=str(ranking_config["left_support_shard_dir"]),
        )
        partial_left = _load_partial_embedding_shards(
            Path(str(ranking_config["left_support_shard_dir"])),
            representations=[left_rep],
            allowed_sample_ids=support_ids_allowed,
            max_shards=_optional_int(ranking_config.get("left_support_max_shards")),
        )
        left_support = partial_left["embeddings"][left_rep]
        left_support_ids = list(partial_left["sample_ids"])
        if smoke:
            smoke_support = int(config["execution"].get("smoke_left_support_samples", config["execution"].get("smoke_window_support_samples", 256)))
            left_support = left_support[:smoke_support]
            left_support_ids = left_support_ids[:smoke_support]
        min_left_support = 1 if smoke else int(ranking_config.get("min_left_support_clips", 1))
        if len(left_support) < min_left_support:
            raise ValueError(
                f"Exact-window ranking found only {len(left_support)} cached '{left_rep}' support embeddings; "
                f"required at least {min_left_support}."
            )
        log_event(
            "active_exact_window_blend_rank",
            "left_support_load_done",
            mode=mode,
            representation=left_rep,
            n_left_support=len(left_support),
        )
    else:
        left_support_ids = []
        left_support = np.empty((0, 0), dtype="float32")
        log_event(
            "active_exact_window_blend_rank",
            "left_support_load_skipped",
            mode=mode,
            selector_mode=selector_mode,
            reason="exact-window old-novelty mode uses full window shards only",
        )

    shard_manifest_path = _right_support_manifest_path(ranking_config, mode=mode)
    log_event(
        "active_exact_window_blend_rank",
        "right_window_shards_load_start",
        mode=mode,
        representation=right_rep,
        manifest_path=str(shard_manifest_path),
    )
    shard_lookup = _load_window_shard_embeddings(
        shard_manifest_path,
        representation=right_rep,
        support_split=support_split,
        query_split=query_split,
    )
    right_support_ids = shard_lookup["support_ids"]
    right_support = shard_lookup["support_embeddings"]
    if smoke:
        smoke_support = int(config["execution"].get("smoke_window_support_samples", 256))
        right_support_ids = right_support_ids[:smoke_support]
        right_support = right_support[:smoke_support]
    query_ids = [clip.sample_id for clip in query_clips]
    right_query = _matrix_for_ids(
        shard_lookup["query_ids"],
        shard_lookup["query_embeddings"],
        query_ids,
        label=f"{right_rep} query shard",
    )
    log_event(
        "active_exact_window_blend_rank",
        "right_window_shards_load_done",
        mode=mode,
        representation=right_rep,
        n_right_support=len(right_support),
        n_right_query=len(right_query),
    )

    old_k_right = min(max(1, int(ranking_config.get("old_novelty_k", 10))), len(right_support_ids))
    if uses_left_cache:
        left_query_shard_dir = _left_query_shard_dir(ranking_config, mode=mode)
        log_event(
            "active_exact_window_blend_rank",
            "left_query_load_start",
            mode=mode,
            representation=left_rep,
            n_query=len(query_clips),
            shard_dir=str(left_query_shard_dir),
        )
        partial_query = _load_partial_embedding_shards(
            left_query_shard_dir,
            representations=[left_rep],
            allowed_sample_ids=set(query_ids),
            max_shards=_optional_int(ranking_config.get("left_query_max_shards")),
        )
        left_query = _matrix_for_ids(
            partial_query["sample_ids"],
            partial_query["embeddings"][left_rep],
            query_ids,
            label=f"{left_rep} query shard",
        )
        query_cache_report = {
            "status": "partial_shard_hit",
            "path": str(left_query_shard_dir),
            "n_clips": int(len(partial_query["sample_ids"])),
        }
        log_event(
            "active_exact_window_blend_rank",
            "left_query_load_done",
            mode=mode,
            representation=left_rep,
            n_query=len(left_query),
            cache_report=query_cache_report,
        )
        old_k_left = min(max(1, int(ranking_config.get("old_novelty_k", 10))), len(left_support_ids))
    else:
        left_query = right_query
        old_k_left = 0
        query_cache_report = {
            "status": "full_support_shard_hit",
            "path": str(shard_manifest_path),
            "n_clips": int(len(right_query)),
        }
    log_event(
        "active_exact_window_blend_rank",
        "novelty_compute_start",
        mode=mode,
        old_k_left=old_k_left,
        old_k_right=old_k_right,
        n_left_support=len(left_support),
        n_right_support=len(right_support),
        n_query=len(query_clips),
    )
    _right_order, right_novelty = old_novelty_only_order(right_support, right_query, k=old_k_right)
    if uses_left_cache:
        _left_order, left_novelty = old_novelty_only_order(left_support, left_query, k=old_k_left)
    else:
        left_novelty = right_novelty
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
    log_event(
        "active_exact_window_blend_rank",
        "novelty_compute_done",
        mode=mode,
        n_rows=len(rows),
        n_clusters=len(set(int(cluster_id) for cluster_id in cluster_ids.tolist())),
    )

    alpha = float(ranking_config.get("alpha", 0.5))
    quality_threshold = float(ranking_config.get("quality_threshold", 0.85))
    max_stationary_fraction = _optional_float(ranking_config.get("max_stationary_fraction", 0.90))
    max_abs_value = _optional_float(ranking_config.get("max_abs_value", 60.0))
    if selector_mode == "old_novelty_only":
        order = _quality_gated_old_novelty_order(
            rows,
            right_novelty,
            quality_threshold=quality_threshold,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
        selector = f"exact_window_old_novelty_{right_rep}"
        ranking_mode = "exact_window_old_novelty_only"
    else:
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
        selector = f"exact_window_{_selector_name(left_rep, right_rep, alpha)}"
        ranking_mode = "partial_left_exact_window_right"
    ranked_rows = _ranked_rows(rows, order, selector_name=selector)
    log_event(
        "active_exact_window_blend_rank",
        "ranking_done",
        mode=mode,
        n_ranked=len(ranked_rows),
        selector=selector,
    )

    submission_path = output_dir / f"active_exact_window_blend_submission_{mode}.csv"
    diagnostics_path = output_dir / f"active_exact_window_blend_diagnostics_{mode}.csv"
    report_path = output_dir / f"active_exact_window_blend_report_{mode}.json"
    finalized_worker_id_path = output_dir / f"active_exact_window_blend_submission_{mode}_worker_id.csv"
    finalized_new_worker_id_path = output_dir / f"active_exact_window_blend_submission_{mode}_new_worker_id.csv"

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

    topk_values = [int(k) for k in ranking_config.get("top_k_values", DEFAULT_TOP_K_VALUES)]
    comparison = _comparison_report(
        ranked_rows,
        config.get("comparison", {}),
        k_values=topk_values,
    )
    report = {
        "mode": mode,
        "selector": selector,
        "ranking_mode": ranking_mode,
        "support_split": support_split,
        "query_split": query_split,
        "n_left_support": int(len(left_support)),
        "n_right_support": int(len(right_support)),
        "n_query": int(len(query_clips)),
        "left_representation": left_rep,
        "right_representation": right_rep,
        "alpha": alpha,
        "old_novelty_k_left": old_k_left,
        "old_novelty_k_right": old_k_right,
        "quality_threshold": quality_threshold,
        "max_stationary_fraction": max_stationary_fraction,
        "max_abs_value": max_abs_value,
        "left_support_cache": {
            "status": "partial_shard_hit" if uses_left_cache else "not_used",
            "path": str(ranking_config.get("left_support_shard_dir", "")),
            "n_clips": int(len(left_support)),
        },
        "right_support_cache": {
            "status": "full_support_shard_hit",
            "path": str(shard_manifest_path),
            "n_clips": int(len(right_support)),
        },
        "query_embedding_cache": query_cache_report,
        "registry_coverage": _deduplicate_coverage_aliases(registry_coverage),
        "topk_quality": _topk_quality_summary(ranked_rows, k_values=topk_values),
        "comparison": comparison,
        "artifacts": {
            "submission": str(submission_path),
            "diagnostics": str(diagnostics_path),
            "finalized_worker_id": str(finalized_worker_id_path),
            "finalized_new_worker_id": str(finalized_new_worker_id_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log_event(
        "active_exact_window_blend_rank",
        "artifacts_written",
        mode=mode,
        submission_path=str(submission_path),
        diagnostics_path=str(diagnostics_path),
        report_path=str(report_path),
    )
    result = {
        "mode": mode,
        "selector": report["selector"],
        "ranking_mode": ranking_mode,
        "n_left_support": int(len(left_support)),
        "n_right_support": int(len(right_support)),
        "n_query": int(len(query_clips)),
        "submission_path": str(submission_path),
        "diagnostics_path": str(diagnostics_path),
        "finalized_worker_id_path": str(finalized_worker_id_path),
        "finalized_new_worker_id_path": str(finalized_new_worker_id_path),
        "report_path": str(report_path),
    }
    log_event("active_exact_window_blend_rank", "done", **result)
    return result


def validate_active_exact_window_blend_rank_config(config: Mapping[str, Any]) -> None:
    execution = _required_mapping(config, "execution")
    data = _required_mapping(config, "data")
    artifacts = _required_mapping(config, "artifacts")
    ranking = _required_mapping(config, "ranking")
    manifests = _required_mapping(data, "manifests")
    if execution.get("provider") != "modal":
        raise ValueError("Exact-window blend ranking must run on Modal.")
    allow_local_paths = bool(execution.get("allow_local_paths_for_tests", False))
    if not allow_local_paths and not str(data.get("root", "")).startswith("/data"):
        raise ValueError("data.root must be mounted under /data.")
    if not allow_local_paths and not str(artifacts.get("output_dir", "")).startswith("/artifacts"):
        raise ValueError("artifacts.output_dir must be mounted under /artifacts.")
    for split in (str(ranking.get("support_split", "pretrain")), str(ranking.get("query_split", "new"))):
        if split not in manifests:
            raise ValueError(f"data.manifests must include split '{split}'.")
    left_rep = str(ranking.get("left_representation", ""))
    right_rep = str(ranking.get("right_representation", "window_mean_std_pool"))
    selector_mode = str(ranking.get("selector_mode", "blend_kcenter"))
    if selector_mode not in {"blend_kcenter", "old_novelty_only"}:
        raise ValueError("ranking.selector_mode must be blend_kcenter or old_novelty_only.")
    unsupported = {left_rep, right_rep} - set(SUPPORTED_REPRESENTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported exact-window blend representations: {sorted(unsupported)}")
    if right_rep != "window_mean_std_pool":
        raise ValueError("ranking.right_representation must be window_mean_std_pool.")
    if selector_mode == "old_novelty_only" and left_rep != "window_mean_std_pool":
        raise ValueError("ranking.left_representation must be window_mean_std_pool in old_novelty_only mode.")
    if selector_mode != "old_novelty_only":
        if not str(ranking.get("left_support_shard_dir", "")).strip():
            raise ValueError("ranking.left_support_shard_dir is required.")
        if not any(str(ranking.get(key, "")).strip() for key in ("left_query_shard_dir", "candidate_query_shard_dir", "candidate_cache_shard_dir")):
            raise ValueError("ranking.left_query_shard_dir is required for cached query embeddings.")
    if not str(ranking.get("right_support_shard_manifest", ranking.get("right_support_shard_dir", ""))).strip():
        raise ValueError("ranking.right_support_shard_manifest or ranking.right_support_shard_dir is required.")
    if int(ranking.get("min_left_support_clips", 1)) <= 0:
        raise ValueError("ranking.min_left_support_clips must be positive.")
    if int(ranking.get("old_novelty_k", 10)) <= 0:
        raise ValueError("ranking.old_novelty_k must be positive.")
    alpha = float(ranking.get("alpha", 0.5))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("ranking.alpha must be in [0, 1].")
    if ranking.get("max_query_clips") is not None and int(ranking["max_query_clips"]) <= 0:
        raise ValueError("ranking.max_query_clips must be positive when provided.")
    for value in ranking.get("top_k_values", DEFAULT_TOP_K_VALUES):
        if int(value) <= 0:
            raise ValueError("ranking.top_k_values must contain positive integers.")


def _load_window_shard_embeddings(
    manifest_path: Path,
    *,
    representation: str,
    support_split: str,
    query_split: str,
) -> dict[str, object]:
    manifest = load_shard_manifest(manifest_path)
    key = f"rep__{representation}"
    split_ids: dict[str, list[str]] = {support_split: [], query_split: []}
    split_matrices: dict[str, list[np.ndarray]] = {support_split: [], query_split: []}
    shard_entries = manifest.get("shards", [])
    total_shards = len(shard_entries) if isinstance(shard_entries, list) else 0
    log_event(
        "active_exact_window_blend_rank",
        "right_window_shard_manifest_loaded",
        manifest_path=str(manifest_path),
        n_shards=total_shards,
        representation=representation,
    )
    for shard_index, shard in enumerate(iter_shard_arrays(manifest), start=1):
        if key not in shard:
            raise KeyError(f"Full-support shard is missing representation '{representation}'.")
        splits = np.asarray(shard["splits"]).astype(str)
        sample_ids = np.asarray(shard["sample_ids"]).astype(str)
        matrix = np.asarray(shard[key], dtype="float32")
        for split in (support_split, query_split):
            indices = np.flatnonzero(splits == split)
            if len(indices) == 0:
                continue
            split_ids[split].extend(str(sample_id) for sample_id in sample_ids[indices].tolist())
            split_matrices[split].append(matrix[indices])
        if total_shards and (shard_index == total_shards or shard_index % 10 == 0):
            log_event(
                "active_exact_window_blend_rank",
                "right_window_shard_load_progress",
                index=shard_index,
                total=total_shards,
                support_rows=len(split_ids[support_split]),
                query_rows=len(split_ids[query_split]),
            )
    if not split_matrices[support_split]:
        raise ValueError(f"Full-support shard manifest has no rows for support split '{support_split}'.")
    if not split_matrices[query_split]:
        raise ValueError(f"Full-support shard manifest has no rows for query split '{query_split}'.")
    return {
        "support_ids": split_ids[support_split],
        "support_embeddings": np.vstack(split_matrices[support_split]).astype("float32"),
        "query_ids": split_ids[query_split],
        "query_embeddings": np.vstack(split_matrices[query_split]).astype("float32"),
    }


def _matrix_for_ids(all_ids: Sequence[str], matrix: np.ndarray, selected_ids: Sequence[str], *, label: str) -> np.ndarray:
    by_id = {sample_id: index for index, sample_id in enumerate(all_ids)}
    missing = [sample_id for sample_id in selected_ids if sample_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing {label} rows for {len(missing)} sample IDs: {preview}")
    return np.vstack([matrix[by_id[sample_id]] for sample_id in selected_ids]).astype("float32")


def _right_support_manifest_path(ranking_config: Mapping[str, Any], *, mode: str) -> Path:
    mode_key = f"{mode}_right_support_shard_manifest"
    if ranking_config.get(mode_key):
        return Path(str(ranking_config[mode_key]))
    if ranking_config.get("right_support_shard_manifest"):
        return Path(str(ranking_config["right_support_shard_manifest"]))
    dir_key = f"{mode}_right_support_shard_dir"
    shard_dir = Path(str(ranking_config.get(dir_key, ranking_config["right_support_shard_dir"])))
    return shard_dir / f"full_support_shards_{mode}.json"


def _left_query_shard_dir(ranking_config: Mapping[str, Any], *, mode: str) -> Path:
    mode_key = f"{mode}_left_query_shard_dir"
    if ranking_config.get(mode_key):
        return Path(str(ranking_config[mode_key]))
    if ranking_config.get("left_query_shard_dir"):
        return Path(str(ranking_config["left_query_shard_dir"]))
    if ranking_config.get("candidate_query_shard_dir"):
        return Path(str(ranking_config["candidate_query_shard_dir"]))
    if ranking_config.get("candidate_cache_shard_dir"):
        return Path(str(ranking_config["candidate_cache_shard_dir"]))
    raise ValueError(
        "Exact-window ranking requires ranking.left_query_shard_dir pointing at cached query TS2Vec shards. "
        "This CPU/IO path intentionally does not recompute TS2Vec."
    )


def _apply_full_query_cap(clips: Sequence[object], ranking_config: Mapping[str, Any]) -> list[object]:
    max_query_clips = ranking_config.get("max_query_clips")
    if max_query_clips is None:
        return list(clips)
    max_count = int(max_query_clips)
    if len(clips) > max_count and bool(ranking_config.get("fail_if_query_exceeds_max", True)):
        raise ValueError(
            f"Query split has {len(clips)} clips, exceeding ranking.max_query_clips={max_count}."
        )
    return list(clips[:max_count])


def _quality_gated_old_novelty_order(
    rows: Sequence[Mapping[str, object]],
    novelty: np.ndarray,
    *,
    quality_threshold: float,
    max_stationary_fraction: float | None,
    max_abs_value: float | None,
) -> list[int]:
    eligible: list[int] = []
    fallback: list[int] = []
    for idx, row in enumerate(rows):
        quality_pass = _safe_float(row.get("quality_score", 0.0)) >= float(quality_threshold)
        stationary_pass = (
            max_stationary_fraction is None
            or _safe_float(row.get("stationary_fraction", 0.0)) <= float(max_stationary_fraction)
        )
        abs_value_pass = max_abs_value is None or _safe_float(row.get("max_abs_value", 0.0)) <= float(max_abs_value)
        if quality_pass and stationary_pass and abs_value_pass:
            eligible.append(int(idx))
        else:
            fallback.append(int(idx))
    ranked_eligible = sorted(eligible, key=lambda idx: (-float(novelty[int(idx)]), _sample_key(rows[int(idx)], int(idx))))
    ranked_fallback = sorted(
        fallback,
        key=lambda idx: (
            -_safe_float(rows[int(idx)].get("quality_score", 0.0)),
            -float(novelty[int(idx)]),
            _sample_key(rows[int(idx)], int(idx)),
        ),
    )
    return [*ranked_eligible, *ranked_fallback]


def _comparison_report(
    ranked_rows: Sequence[Mapping[str, object]],
    comparison_config: object,
    *,
    k_values: Sequence[int],
) -> dict[str, object]:
    if not isinstance(comparison_config, Mapping):
        return {}
    path_value = comparison_config.get("baseline_diagnostics_path")
    if not path_value:
        return {}
    baseline_path = Path(str(path_value))
    if not baseline_path.exists():
        return {"baseline_diagnostics_path": str(baseline_path), "status": "missing"}
    baseline_rows = _read_csv_rows(baseline_path)
    current_ids = [str(row["sample_id"]) for row in ranked_rows]
    baseline_ids = [str(row.get("sample_id", row.get("worker_id", ""))) for row in baseline_rows]
    output: dict[str, object] = {
        "baseline_diagnostics_path": str(baseline_path),
        "status": "compared",
        "rank_spearman": _rank_spearman(current_ids, baseline_ids),
    }
    for k in k_values:
        current_top = set(current_ids[: min(int(k), len(current_ids))])
        baseline_top = set(baseline_ids[: min(int(k), len(baseline_ids))])
        output[f"top{int(k)}_overlap"] = _fraction(len(current_top & baseline_top), min(int(k), len(current_top), len(baseline_top)))
    return output


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rank_spearman(left_ids: Sequence[str], right_ids: Sequence[str]) -> float:
    common = [sample_id for sample_id in left_ids if sample_id in set(right_ids)]
    if len(common) < 2:
        return 0.0
    left_rank = {sample_id: rank for rank, sample_id in enumerate(left_ids, start=1)}
    right_rank = {sample_id: rank for rank, sample_id in enumerate(right_ids, start=1)}
    left = np.asarray([left_rank[sample_id] for sample_id in common], dtype=float)
    right = np.asarray([right_rank[sample_id] for sample_id in common], dtype=float)
    left -= float(np.mean(left))
    right -= float(np.mean(right))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float((left @ right) / denom) if denom > 0 else 0.0


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _sample_key(row: Mapping[str, object], idx: int) -> tuple[str, int]:
    return (str(row.get("sample_id", row.get("worker_id", ""))), int(idx))


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"active exact-window blend config requires object field '{key}'.")
    return value
