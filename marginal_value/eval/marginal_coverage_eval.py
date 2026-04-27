from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import numpy as np

from marginal_value.data.split_manifest import SplitSample, build_split_manifest, select_split
from marginal_value.indexing.knn_features import normalize_rows
from marginal_value.logging_utils import log_event, log_progress
from marginal_value.preprocessing.quality import load_modal_jsonl_imu, quality_scores_for_rows
from marginal_value.ranking.baseline_ranker import (
    annotate_cluster_features,
    batch_density,
    build_scored_rows,
    compute_batch_clusters,
    old_knn_novelty,
    quality_gated_old_novelty_rank_rows,
    raw_shape_stats_embedding,
    temporal_order_embedding,
    window_mean_std_embedding,
)
from marginal_value.ranking.modal_baseline_rank import (
    GRAMMAR_FEATURE_COLUMNS,
    _join_grammar_features,
    _join_quality_metadata,
    _load_grammar_features,
    _maybe_apply_score_guards,
    _maybe_promote_grammar_scores,
    _maybe_split_large_clusters,
    _rank_rows,
)


def run_marginal_coverage_eval(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Measure whether selected candidates improve coverage of held-out targets.

    This eval is deliberately different from candidate precision. It withholds
    source groups from old support, splits those held-out rows into candidate
    and target sets, ranks the candidates, then asks whether adding top-K
    candidates reduces target nearest-neighbor distance more than simple
    baselines.
    """

    mode = "smoke" if smoke else "full"
    log_event("marginal_coverage_eval", "start", mode=mode)
    rng = np.random.default_rng(int(config.get("seed", config.get("ranking", {}).get("seed", 17))))
    data_root = Path(config["data"]["root"])
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    grammar_features, grammar_feature_path = _load_grammar_features(config, mode=mode)

    manifest = build_split_manifest(
        data_root,
        pretrain_manifest=config["data"]["pretrain_manifest"],
        val_manifest=config["data"]["val_manifest"],
        extra_manifests={},
        feature_glob=config["data"].get("feature_glob", "cache/features/*.npz"),
        raw_glob=config["data"].get("raw_glob", "cache/raw/*.jsonl"),
    )
    rows = select_split(manifest, "pretrain")
    max_rows = int(config["eval"].get("smoke_max_rows" if smoke else "max_rows", len(rows)))
    rows = _sample_rows(rows, max_rows=max_rows, rng=rng)
    source_groups = np.asarray([_source_group_id(row.url) for row in rows], dtype=object)
    log_event(
        "marginal_coverage_eval",
        "rows_ready",
        n_rows=len(rows),
        n_source_groups=len(set(source_groups.tolist())),
        max_rows=max_rows,
    )

    representations = [str(rep) for rep in config["eval"].get("representations", ["window_mean_std_pool"])]
    bundle = _load_embedding_bundle(
        rows,
        representations=representations,
        grammar_features=grammar_features,
        sample_rate=float(config.get("quality", {}).get("sample_rate", 30.0)),
        max_raw_samples=_max_raw_samples(config),
        workers=int(config["eval"].get("embedding_load_workers", 8)),
    )
    folds = _select_source_group_folds(
        source_groups,
        n_folds=int(config["eval"].get("smoke_folds" if smoke else "folds", 4)),
        source_groups_per_fold=int(config["eval"].get("source_groups_per_fold", 64)),
    )
    if not folds:
        raise ValueError("Marginal coverage eval requires at least one source-group fold.")

    fold_reports: list[dict[str, Any]] = []
    all_candidate_rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        fold_report, candidate_rows = _run_marginal_coverage_fold(
            rows=rows,
            source_groups=source_groups,
            heldout_source_groups=[str(group) for group in fold["heldout_source_groups"]],
            ranker_embeddings=bundle.ranker_embeddings,
            eval_embeddings=bundle.eval_embeddings,
            config=config,
            rng=rng,
            fold_index=fold_index,
            grammar_features=grammar_features,
        )
        fold_reports.append(fold_report)
        all_candidate_rows.extend(candidate_rows)
        log_event(
            "marginal_coverage_eval",
            "fold_done",
            fold=fold_index,
            n_candidate=fold_report["n_candidate"],
            n_target=fold_report["n_target"],
        )

    report = {
        "mode": mode,
        "n_rows": len(rows),
        "n_source_groups": int(len(set(source_groups.tolist()))),
        "representations": representations,
        "folds": fold_reports,
        "mean_coverage": _mean_coverage_report(fold_reports),
        "grammar_features": {
            "enabled": bool(config.get("grammar_features", {}).get("enabled", False)),
            "source_path": str(grammar_feature_path) if grammar_feature_path is not None else None,
            "n_rows": len(grammar_features),
        },
        "config": {
            "pretrain_manifest": config["data"]["pretrain_manifest"],
            "max_rows": max_rows,
            "folds": len(fold_reports),
        },
    }
    report_path = output_dir / f"marginal_coverage_report_{mode}.json"
    candidates_path = output_dir / f"marginal_coverage_candidates_{mode}.jsonl"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in all_candidate_rows:
            handle.write(json.dumps(_json_safe_row(row), sort_keys=True) + "\n")

    result = {
        "mode": mode,
        "n_rows": len(rows),
        "n_source_groups": report["n_source_groups"],
        "n_folds": len(fold_reports),
        "report_path": str(report_path),
        "candidates_path": str(candidates_path),
    }
    log_event("marginal_coverage_eval", "done", **result)
    return result


def coverage_gain_for_selection(
    support_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    *,
    selected_indices: Sequence[int],
) -> dict[str, float]:
    support = np.asarray(support_embeddings, dtype=float)
    target = np.asarray(target_embeddings, dtype=float)
    candidates = np.asarray(candidate_embeddings, dtype=float)
    indices = [int(idx) for idx in selected_indices]
    baseline_distance = _mean_nearest_distance(target, support)
    if indices:
        selected = candidates[np.asarray(indices, dtype=int)]
        augmented = np.vstack([support, selected])
    else:
        augmented = support
    after_distance = _mean_nearest_distance(target, augmented)
    gain = float(baseline_distance - after_distance)
    relative = float(gain / baseline_distance) if baseline_distance > 1.0e-12 else 0.0
    return {
        "baseline_distance": float(baseline_distance),
        "after_distance": float(after_distance),
        "coverage_gain": gain,
        "relative_coverage_gain": relative,
        "selected_count": float(len(indices)),
    }


class _EmbeddingBundle:
    def __init__(self, *, ranker_embeddings: np.ndarray, eval_embeddings: dict[str, np.ndarray]) -> None:
        self.ranker_embeddings = ranker_embeddings
        self.eval_embeddings = eval_embeddings


def _run_marginal_coverage_fold(
    *,
    rows: list[SplitSample],
    source_groups: np.ndarray,
    heldout_source_groups: list[str],
    ranker_embeddings: np.ndarray,
    eval_embeddings: dict[str, np.ndarray],
    config: dict[str, Any],
    rng: np.random.Generator,
    fold_index: int,
    grammar_features: dict[str, dict[str, object]],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    heldout_mask = np.isin(source_groups, np.asarray(heldout_source_groups, dtype=object))
    positive_candidate_indices, target_indices = _split_heldout_candidate_target(
        source_groups,
        heldout_source_groups=heldout_source_groups,
        candidate_fraction=float(config["eval"].get("candidate_fraction", 0.5)),
        rng=rng,
    )
    positive_candidate_indices = _sample_indices(
        positive_candidate_indices,
        max_size=int(config["eval"].get("max_candidate_per_fold", len(positive_candidate_indices))),
        rng=rng,
    )
    target_indices = _sample_indices(
        target_indices,
        max_size=int(config["eval"].get("max_target_per_fold", len(target_indices))),
        rng=rng,
    )
    negative_pool = np.flatnonzero(~heldout_mask)
    negative_indices = _sample_indices(
        negative_pool,
        max_size=int(config["eval"].get("max_negative_per_fold", len(negative_pool))),
        rng=rng,
    )
    candidate_indices = np.concatenate([positive_candidate_indices, negative_indices])
    if len(candidate_indices) == 0 or len(target_indices) == 0:
        raise ValueError("Each marginal coverage fold requires candidates and held-out targets.")

    support_mask = ~heldout_mask
    support_mask[negative_indices] = False
    support_indices = np.flatnonzero(support_mask)
    if len(support_indices) == 0:
        raise ValueError("Each marginal coverage fold requires non-candidate support rows.")

    candidate_rows = [rows[int(idx)] for idx in candidate_indices]
    candidate_embeddings = ranker_embeddings[candidate_indices]
    support_embeddings = ranker_embeddings[support_indices]
    old_distance, _neighbors = old_knn_novelty(
        support_embeddings,
        candidate_embeddings,
        k=int(config["ranking"].get("k_old", config["eval"].get("k_old", 10))),
    )
    density = batch_density(
        candidate_embeddings,
        k=int(config["ranking"].get("k_new_density", config["eval"].get("k_new_density", 10))),
    )
    quality_scores, quality_metadata = quality_scores_for_rows(
        candidate_rows,
        sample_rate=float(config.get("quality", {}).get("sample_rate", 30.0)),
        max_samples=config.get("quality", {}).get("max_samples_per_clip"),
        log_component="marginal_coverage_eval",
        log_label=f"fold_{fold_index}_candidate",
    )
    scored = build_scored_rows(
        sample_ids=[row.sample_id for row in candidate_rows],
        embeddings=candidate_embeddings,
        old_knn_distance=old_distance,
        new_density=density,
        quality_scores=quality_scores,
        novelty_weight=float(config["ranking"].get("novelty_weight", config["eval"].get("novelty_weight", 0.75))),
    )
    scored = _join_quality_metadata(scored, quality_metadata)
    for candidate_index, row in enumerate(scored):
        source_index = int(candidate_indices[candidate_index])
        is_target_source_group = bool(heldout_mask[source_index])
        row["candidate_index"] = int(candidate_index)
        row["source_row_index"] = source_index
        row["fold"] = int(fold_index)
        row["source_group_id"] = str(source_groups[source_index])
        row["is_target_source_group"] = int(is_target_source_group)
        row["candidate_type"] = "heldout_source_candidate" if is_target_source_group else "source_covered_distractor"
        row["heldout_source_groups"] = ",".join(heldout_source_groups)

    scored = _join_grammar_features(scored, grammar_features)
    scored, grammar_summary = _maybe_promote_grammar_scores(config, scored, label=f"marginal_coverage_fold_{fold_index}")
    candidate_cluster_ids = compute_batch_clusters(
        candidate_embeddings,
        similarity_threshold=float(
            config["ranking"].get(
                "cluster_similarity_threshold",
                config["eval"].get("candidate_cluster_similarity_threshold", 0.985),
            )
        ),
    )
    scored = annotate_cluster_features(scored, candidate_embeddings, candidate_cluster_ids)
    scored, score_guards = _maybe_apply_score_guards(config, scored, label=f"marginal_coverage_fold_{fold_index}")
    scored, large_cluster_split = _maybe_split_large_clusters(
        config,
        scored,
        candidate_embeddings,
        label=f"marginal_coverage_fold_{fold_index}",
    )
    ranking_config = config.get("ranking", {})
    ranked = _rank_rows(
        scored,
        candidate_embeddings,
        reranker_method=str(ranking_config.get("reranker_method", "tiered_cluster_cap")),
        mmr_lambda=float(ranking_config.get("mmr_lambda", config["eval"].get("mmr_lambda", 0.0))),
        cluster_bonus_weight=float(ranking_config.get("cluster_bonus_weight", config["eval"].get("cluster_bonus_weight", 0.0))),
        cluster_cap_top_k=int(ranking_config.get("cluster_cap_top_k", config["eval"].get("cluster_cap_top_k", 200))),
        cluster_max_per_cluster=int(ranking_config.get("cluster_max_per_cluster", config["eval"].get("cluster_max_per_cluster", 8))),
        cluster_cap_key=str(ranking_config.get("cluster_cap_key", config["eval"].get("cluster_cap_key", "new_cluster_id"))),
        cluster_cap_min_quality=float(ranking_config.get("cluster_cap_min_quality", config["eval"].get("cluster_cap_min_quality", 0.45))),
        cluster_cap_schedule=ranking_config.get("cluster_cap_schedule"),
        prefix_cluster_cap_top_k=int(ranking_config.get("prefix_cluster_cap_top_k", config["eval"].get("prefix_cluster_cap_top_k", 75))),
        prefix_cluster_cap_key=str(ranking_config.get("prefix_cluster_cap_key", config["eval"].get("prefix_cluster_cap_key", "new_cluster_parent_id"))),
        prefix_cluster_max_per_cluster=int(ranking_config.get("prefix_cluster_max_per_cluster", ranking_config.get("cluster_max_per_cluster", 8))),
        quality_gate_threshold=float(ranking_config.get("quality_gate_threshold", 0.45)),
        source_cap=ranking_config.get("source_cap"),
        source_cap_key=str(ranking_config.get("source_cap_key", "source_group_id")),
    )
    quality_gate_thresholds = _resolve_quality_gate_thresholds(
        config,
        support_rows=[rows[int(idx)] for idx in support_indices],
        sample_rate=float(config.get("quality", {}).get("sample_rate", 30.0)),
        max_samples=config.get("quality", {}).get("max_samples_per_clip"),
        fold_index=fold_index,
    )
    policies = _evaluate_policies(
        scored=scored,
        ranked=ranked,
        candidate_indices=candidate_indices,
        support_indices=support_indices,
        target_indices=target_indices,
        eval_embeddings=eval_embeddings,
        k_values=[int(k) for k in config["eval"].get("k_values", [10, 50, 100, 200])],
        random_seeds=[int(seed) for seed in config["eval"].get("random_seeds", [17, 23, 37, 41, 53])],
        random_min_quality=float(config["eval"].get("random_min_quality", 0.45)),
        quality_gate_thresholds=quality_gate_thresholds,
        quality_gate_source_caps=_quality_gate_source_caps(config),
        quality_gate_source_cap_key=_quality_gate_source_cap_key(config),
        quality_gate_validity_specs=_quality_gate_validity_specs(config),
        eval_rep_novelty_representations=[
            str(rep) for rep in config["eval"].get("eval_rep_novelty_representations", [])
        ],
        eval_rep_novelty_k_old=int(config["eval"].get("eval_rep_novelty_k_old", config["ranking"].get("k_old", 10))),
        quality_gate_eval_rep_novelty=bool(config["eval"].get("quality_gate_eval_rep_novelty", False)),
        quality_gated_random_controls=_quality_gated_random_control_specs(config),
        kcenter_greedy_controls=_kcenter_greedy_control_specs(config),
        cluster_similarity_threshold=float(
            config["ranking"].get(
                "cluster_similarity_threshold",
                config["eval"].get("candidate_cluster_similarity_threshold", 0.985),
            )
        ),
    )

    fold_report = {
        "fold": int(fold_index),
        "heldout_source_groups": heldout_source_groups,
        "n_support": int(len(support_indices)),
        "n_candidate": int(len(candidate_indices)),
        "n_target": int(len(target_indices)),
        "n_target_source_candidate": int(len(positive_candidate_indices)),
        "n_source_covered_distractor": int(len(negative_indices)),
        "target_source_group_count": int(len(set(str(source_groups[int(idx)]) for idx in target_indices))),
        "candidate_source_group_count": int(len(set(str(source_groups[int(idx)]) for idx in candidate_indices))),
        "policies": policies,
        "grammar_promotion": grammar_summary,
        "score_guards": score_guards,
        "large_cluster_split": large_cluster_split,
    }
    return fold_report, ranked


def _evaluate_policies(
    *,
    scored: list[dict[str, object]],
    ranked: list[dict[str, object]],
    candidate_indices: np.ndarray,
    support_indices: np.ndarray,
    target_indices: np.ndarray,
    eval_embeddings: dict[str, np.ndarray],
    k_values: Sequence[int],
    random_seeds: Sequence[int],
    random_min_quality: float,
    quality_gate_thresholds: Sequence[tuple[str, float]],
    quality_gate_source_caps: Sequence[int],
    quality_gate_source_cap_key: str,
    quality_gate_validity_specs: Sequence[dict[str, object]],
    eval_rep_novelty_representations: Sequence[str],
    eval_rep_novelty_k_old: int,
    quality_gate_eval_rep_novelty: bool,
    quality_gated_random_controls: Sequence[dict[str, object]],
    kcenter_greedy_controls: Sequence[dict[str, object]],
    cluster_similarity_threshold: float,
) -> dict[str, dict[str, object]]:
    policy_orders = {
        "ranker": [int(row["candidate_index"]) for row in ranked],
        "quality_only": _order_by_score(scored, ("quality_score", "sample_id")),
        "old_novelty_only": _order_by_score(scored, ("old_novelty_score", "sample_id")),
        "new_density_only": _order_by_score(scored, ("new_density_score", "sample_id")),
        "final_score_only": _order_by_score(scored, ("final_score", "sample_id")),
        "diverse_source_cluster": _diverse_source_cluster_order(scored),
    }
    for spec in kcenter_greedy_controls:
        name = _control_policy_name(spec, default="kcenter_greedy_quality_gated")
        representation = str(spec.get("representation", "window_mean_std_pool"))
        if representation not in eval_embeddings:
            raise ValueError(f"k-center control representation is not loaded: {representation}")
        policy_orders[name] = _kcenter_greedy_quality_gated_order(
            scored=scored,
            candidate_indices=candidate_indices,
            support_indices=support_indices,
            embeddings=eval_embeddings[representation],
            quality_threshold=_control_quality_threshold(spec, default=0.45),
            max_stationary_fraction=_optional_control_float(spec, "max_stationary_fraction"),
            max_abs_value=_optional_control_float(spec, "max_abs_value"),
        )
    for representation in eval_rep_novelty_representations:
        if representation not in eval_embeddings:
            raise ValueError(f"eval_rep_novelty representation is not loaded: {representation}")
        eval_rep_rows = _eval_rep_old_novelty_rows(
            scored=scored,
            candidate_indices=candidate_indices,
            support_indices=support_indices,
            eval_embeddings=eval_embeddings,
            representation=representation,
            k_old=eval_rep_novelty_k_old,
            cluster_similarity_threshold=cluster_similarity_threshold,
        )
        policy_orders[f"old_novelty_{representation}"] = _eval_rep_old_novelty_order(eval_rep_rows)
        if quality_gate_eval_rep_novelty:
            for threshold_name, threshold in quality_gate_thresholds:
                policy_orders[f"quality_gated_old_novelty_{representation}_{threshold_name}"] = (
                    _quality_gated_old_novelty_order(
                        eval_rep_rows,
                        quality_threshold=threshold,
                    )
                )
                for source_cap in quality_gate_source_caps:
                    source_cap_suffix = _source_cap_suffix(source_cap, quality_gate_source_cap_key)
                    policy_orders[
                        f"quality_gated_old_novelty_{representation}_{threshold_name}_{source_cap_suffix}"
                    ] = _quality_gated_old_novelty_order(
                        eval_rep_rows,
                        quality_threshold=threshold,
                        source_cap=int(source_cap),
                        source_key=quality_gate_source_cap_key,
                    )
                for validity_spec in quality_gate_validity_specs:
                    validity_name = str(validity_spec["name"])
                    policy_orders[f"quality_gated_old_novelty_{representation}_{threshold_name}_{validity_name}"] = (
                        _quality_gated_old_novelty_order(
                            eval_rep_rows,
                            quality_threshold=threshold,
                            max_stationary_fraction=validity_spec.get("max_stationary_fraction"),
                            max_abs_value=validity_spec.get("max_abs_value"),
                        )
                    )
                    for source_cap in quality_gate_source_caps:
                        source_cap_suffix = _source_cap_suffix(source_cap, quality_gate_source_cap_key)
                        policy_orders[
                            f"quality_gated_old_novelty_{representation}_{threshold_name}_{validity_name}_{source_cap_suffix}"
                        ] = _quality_gated_old_novelty_order(
                            eval_rep_rows,
                            quality_threshold=threshold,
                            max_stationary_fraction=validity_spec.get("max_stationary_fraction"),
                            max_abs_value=validity_spec.get("max_abs_value"),
                            source_cap=int(source_cap),
                            source_key=quality_gate_source_cap_key,
                        )
    for threshold_name, threshold in quality_gate_thresholds:
        policy_orders[f"quality_gated_old_novelty_{threshold_name}"] = _quality_gated_old_novelty_order(
            scored,
            quality_threshold=threshold,
        )
        for validity_spec in quality_gate_validity_specs:
            validity_name = str(validity_spec["name"])
            policy_orders[f"quality_gated_old_novelty_{threshold_name}_{validity_name}"] = (
                _quality_gated_old_novelty_order(
                    scored,
                    quality_threshold=threshold,
                    max_stationary_fraction=validity_spec.get("max_stationary_fraction"),
                    max_abs_value=validity_spec.get("max_abs_value"),
                )
            )
        for source_cap in quality_gate_source_caps:
            source_cap_suffix = _source_cap_suffix(source_cap, quality_gate_source_cap_key)
            policy_orders[f"quality_gated_old_novelty_{threshold_name}_{source_cap_suffix}"] = (
                _quality_gated_old_novelty_order(
                    scored,
                    quality_threshold=threshold,
                    source_cap=int(source_cap),
                    source_key=quality_gate_source_cap_key,
                )
            )
    reports: dict[str, dict[str, object]] = {}
    for policy_name, order in policy_orders.items():
        reports[policy_name] = _policy_coverage_report(
            scored=scored,
            order=order,
            candidate_indices=candidate_indices,
            support_indices=support_indices,
            target_indices=target_indices,
            eval_embeddings=eval_embeddings,
            k_values=k_values,
        )
    reports["random_high_quality"] = _random_policy_report(
        scored=scored,
        candidate_indices=candidate_indices,
        support_indices=support_indices,
        target_indices=target_indices,
        eval_embeddings=eval_embeddings,
        k_values=k_values,
        seeds=random_seeds,
        min_quality=random_min_quality,
    )
    for spec in quality_gated_random_controls:
        name = _control_policy_name(spec, default="quality_gated_random")
        reports[name] = _random_policy_report(
            scored=scored,
            candidate_indices=candidate_indices,
            support_indices=support_indices,
            target_indices=target_indices,
            eval_embeddings=eval_embeddings,
            k_values=k_values,
            seeds=random_seeds,
            min_quality=_control_quality_threshold(spec, default=random_min_quality),
            max_stationary_fraction=_optional_control_float(spec, "max_stationary_fraction"),
            max_abs_value=_optional_control_float(spec, "max_abs_value"),
            source_cap=_optional_control_int(spec, "source_cap"),
            source_key=str(spec.get("source_cap_key", "source_group_id")),
        )
    return reports


def _policy_coverage_report(
    *,
    scored: list[dict[str, object]],
    order: Sequence[int],
    candidate_indices: np.ndarray,
    support_indices: np.ndarray,
    target_indices: np.ndarray,
    eval_embeddings: dict[str, np.ndarray],
    k_values: Sequence[int],
) -> dict[str, object]:
    report: dict[str, object] = {"order_size": int(len(order))}
    candidate_index_array = np.asarray(candidate_indices, dtype=int)
    for k in k_values:
        selected = [int(idx) for idx in order[: min(int(k), len(order))]]
        coverage_by_rep: dict[str, dict[str, float]] = {}
        for rep_name, embeddings in eval_embeddings.items():
            coverage_by_rep[rep_name] = coverage_gain_for_selection(
                embeddings[support_indices],
                embeddings[target_indices],
                embeddings[candidate_index_array],
                selected_indices=selected,
            )
        report[f"coverage@{int(k)}"] = coverage_by_rep
        report[f"selection@{int(k)}"] = _selection_summary(scored, selected)
    return report


def _random_policy_report(
    *,
    scored: list[dict[str, object]],
    candidate_indices: np.ndarray,
    support_indices: np.ndarray,
    target_indices: np.ndarray,
    eval_embeddings: dict[str, np.ndarray],
    k_values: Sequence[int],
    seeds: Sequence[int],
    min_quality: float,
    max_stationary_fraction: float | None = None,
    max_abs_value: float | None = None,
    source_cap: int | None = None,
    source_key: str = "source_group_id",
) -> dict[str, object]:
    high_quality = [
        idx
        for idx, row in enumerate(scored)
        if _passes_control_gates(
            row,
            quality_threshold=min_quality,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
    ]
    fallback = [idx for idx in range(len(scored)) if idx not in set(high_quality)]
    seed_orders: list[list[int]] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        high = list(rng.permutation(high_quality).astype(int)) if high_quality else []
        if source_cap is not None:
            high = _source_capped_order(high, scored, source_cap=int(source_cap), source_key=source_key)
        low = list(rng.permutation(fallback).astype(int)) if fallback else []
        seed_orders.append([*high, *low])

    report: dict[str, object] = {
        "order_size": int(len(scored)),
        "random_seeds": [int(seed) for seed in seeds],
        "min_quality": float(min_quality),
        "max_stationary_fraction": "" if max_stationary_fraction is None else float(max_stationary_fraction),
        "max_abs_value": "" if max_abs_value is None else float(max_abs_value),
        "source_cap": "" if source_cap is None else int(source_cap),
        "source_key": "" if source_cap is None else source_key,
    }
    candidate_index_array = np.asarray(candidate_indices, dtype=int)
    for k in k_values:
        selected_by_seed = [order[: min(int(k), len(order))] for order in seed_orders]
        coverage_by_rep: dict[str, dict[str, float]] = {}
        for rep_name, embeddings in eval_embeddings.items():
            seed_metrics = [
                coverage_gain_for_selection(
                    embeddings[support_indices],
                    embeddings[target_indices],
                    embeddings[candidate_index_array],
                    selected_indices=selected,
                )
                for selected in selected_by_seed
            ]
            coverage_by_rep[rep_name] = _mean_std_metrics(seed_metrics)
        report[f"coverage@{int(k)}"] = coverage_by_rep
        report[f"selection@{int(k)}"] = _mean_selection_summary(scored, selected_by_seed)
    return report


def _load_embedding_bundle(
    rows: list[SplitSample],
    *,
    representations: Sequence[str],
    grammar_features: dict[str, dict[str, object]],
    sample_rate: float,
    max_raw_samples: int | None,
    workers: int,
) -> _EmbeddingBundle:
    window_matrices = _load_window_matrices(rows, workers=workers)
    ranker_embeddings = np.vstack([window_mean_std_embedding(windows) for windows in window_matrices])
    eval_embeddings: dict[str, np.ndarray] = {}
    for representation in representations:
        if representation == "window_mean_std_pool":
            eval_embeddings[representation] = ranker_embeddings
        elif representation == "temporal_order":
            eval_embeddings[representation] = np.vstack([temporal_order_embedding(windows) for windows in window_matrices])
        elif representation == "window_shape_stats":
            eval_embeddings[representation] = np.vstack([_window_shape_stats_embedding(windows) for windows in window_matrices])
        elif representation == "raw_shape_stats":
            eval_embeddings[representation] = _load_raw_shape_embeddings(
                rows,
                sample_rate=sample_rate,
                max_samples=max_raw_samples,
                workers=workers,
            )
        elif representation == "grammar_features":
            eval_embeddings[representation] = _grammar_feature_embeddings(rows, grammar_features)
        else:
            raise ValueError(f"Unsupported marginal coverage representation: {representation}")
        log_event(
            "marginal_coverage_eval",
            "representation_ready",
            representation=representation,
            n_rows=int(eval_embeddings[representation].shape[0]),
            n_features=int(eval_embeddings[representation].shape[1]),
        )
    return _EmbeddingBundle(ranker_embeddings=ranker_embeddings, eval_embeddings=eval_embeddings)


def _load_window_matrices(rows: list[SplitSample], *, workers: int) -> list[np.ndarray]:
    total = len(rows)
    if total == 0:
        raise ValueError("No rows available for marginal coverage eval.")
    workers = max(1, int(workers))
    log_event("marginal_coverage_eval", "window_load_start", total=total, workers=workers)
    progress_every = max(1, total // 20)
    matrices: list[np.ndarray | None] = [None] * total
    if workers == 1:
        for idx, row in enumerate(rows, start=1):
            matrices[idx - 1] = _load_one_window_matrix(row)
            log_progress("marginal_coverage_eval", "window_load_progress", index=idx, total=total, every=progress_every)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = {executor.submit(_load_one_window_matrix, row): idx for idx, row in enumerate(rows)}
            for completed, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                matrices[idx] = future.result()
                log_progress(
                    "marginal_coverage_eval",
                    "window_load_progress",
                    index=completed,
                    total=total,
                    every=progress_every,
                )
    loaded = [matrix for matrix in matrices if matrix is not None]
    if len(loaded) != total:
        raise RuntimeError(f"Loaded {len(loaded)} window matrices for {total} rows.")
    log_event("marginal_coverage_eval", "window_load_done", total=total)
    return loaded


def _load_one_window_matrix(row: SplitSample) -> np.ndarray:
    with np.load(row.feature_path) as data:
        return np.asarray(data["window_features"], dtype="float32")


def _load_raw_shape_embeddings(
    rows: list[SplitSample],
    *,
    sample_rate: float,
    max_samples: int | None,
    workers: int,
) -> np.ndarray:
    total = len(rows)
    workers = max(1, int(workers))
    log_event("marginal_coverage_eval", "raw_shape_load_start", total=total, workers=workers, max_samples=max_samples)
    progress_every = max(1, total // 20)
    embeddings: list[np.ndarray | None] = [None] * total
    if workers == 1:
        for idx, row in enumerate(rows, start=1):
            embeddings[idx - 1] = _load_one_raw_shape_embedding(row, sample_rate=sample_rate, max_samples=max_samples)
            log_progress("marginal_coverage_eval", "raw_shape_load_progress", index=idx, total=total, every=progress_every)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = {
                executor.submit(_load_one_raw_shape_embedding, row, sample_rate=sample_rate, max_samples=max_samples): idx
                for idx, row in enumerate(rows)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                idx = futures[future]
                embeddings[idx] = future.result()
                log_progress(
                    "marginal_coverage_eval",
                    "raw_shape_load_progress",
                    index=completed,
                    total=total,
                    every=progress_every,
                )
    loaded = [embedding for embedding in embeddings if embedding is not None]
    if len(loaded) != total:
        raise RuntimeError(f"Loaded {len(loaded)} raw-shape embeddings for {total} rows.")
    log_event("marginal_coverage_eval", "raw_shape_load_done", total=total)
    return np.vstack(loaded)


def _load_one_raw_shape_embedding(row: SplitSample, *, sample_rate: float, max_samples: int | None) -> np.ndarray:
    samples, _timestamps = load_modal_jsonl_imu(row.raw_path, max_samples=max_samples)
    return raw_shape_stats_embedding(samples, sample_rate=sample_rate)


def _window_shape_stats_embedding(windows: np.ndarray) -> np.ndarray:
    values = _finite_2d(windows)
    if len(values) > 1:
        diffs = np.diff(values, axis=0)
        diff_parts = [np.mean(np.abs(diffs), axis=0), np.max(np.abs(diffs), axis=0)]
    else:
        diff_parts = [np.zeros(values.shape[1]), np.zeros(values.shape[1])]
    return np.nan_to_num(
        np.concatenate(
            [
                np.mean(values, axis=0),
                np.std(values, axis=0),
                np.min(values, axis=0),
                np.max(values, axis=0),
                np.percentile(values, 25, axis=0),
                np.percentile(values, 75, axis=0),
                *diff_parts,
            ]
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _grammar_feature_embeddings(
    rows: list[SplitSample],
    grammar_features: dict[str, dict[str, object]],
) -> np.ndarray:
    if not rows:
        return np.empty((0, len(GRAMMAR_FEATURE_COLUMNS)), dtype=float)
    embeddings = []
    for row in rows:
        features = grammar_features.get(row.sample_id, {})
        embeddings.append([_safe_float(features.get(column, 0.0)) for column in GRAMMAR_FEATURE_COLUMNS])
    return np.asarray(embeddings, dtype=float)


def _mean_nearest_distance(query_embeddings: np.ndarray, support_embeddings: np.ndarray) -> float:
    query = normalize_rows(np.asarray(query_embeddings, dtype=float))
    support = normalize_rows(np.asarray(support_embeddings, dtype=float))
    if len(query) == 0:
        raise ValueError("Coverage distance requires at least one target row.")
    if len(support) == 0:
        raise ValueError("Coverage distance requires at least one support row.")
    similarities = query @ support.T
    nearest = np.max(similarities, axis=1)
    distances = 1.0 - nearest
    return float(np.mean(np.maximum(distances, 0.0)))


def _select_source_group_folds(
    source_groups: np.ndarray,
    *,
    n_folds: int,
    source_groups_per_fold: int,
) -> list[dict[str, object]]:
    group_counts = Counter(str(group) for group in source_groups.tolist())
    ordered_groups = [group for group, _count in sorted(group_counts.items(), key=lambda item: (-item[1], item[0]))]
    folds: list[dict[str, object]] = []
    cursor = 0
    groups_per_fold = max(1, int(source_groups_per_fold))
    for _fold in range(max(1, int(n_folds))):
        groups = ordered_groups[cursor : cursor + groups_per_fold]
        if not groups:
            break
        folds.append({"heldout_source_groups": groups})
        cursor += groups_per_fold
    return folds


def _split_heldout_candidate_target(
    source_groups: np.ndarray,
    *,
    heldout_source_groups: Sequence[str],
    candidate_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < candidate_fraction < 1.0:
        raise ValueError("candidate_fraction must be in (0, 1).")
    candidate_indices: list[int] = []
    target_indices: list[int] = []
    for group in heldout_source_groups:
        group_indices = np.flatnonzero(source_groups == str(group))
        if len(group_indices) < 2:
            continue
        shuffled = rng.permutation(group_indices)
        n_candidate = int(np.floor(len(shuffled) * candidate_fraction))
        n_candidate = min(max(1, n_candidate), len(shuffled) - 1)
        candidate_indices.extend(int(idx) for idx in shuffled[:n_candidate])
        target_indices.extend(int(idx) for idx in shuffled[n_candidate:])
    return np.asarray(candidate_indices, dtype=int), np.asarray(target_indices, dtype=int)


def _resolve_quality_gate_thresholds(
    config: dict[str, Any],
    *,
    support_rows: list[SplitSample],
    sample_rate: float,
    max_samples: int | None,
    fold_index: int,
) -> list[tuple[str, float]]:
    gate_config = config.get("eval", {}).get("quality_gated_old_novelty", {})
    if not isinstance(gate_config, dict) or not bool(gate_config.get("enabled", False)):
        return []
    specs = gate_config.get("thresholds", [{"name": "q45", "mode": "fixed", "value": 0.45}])
    if not isinstance(specs, list) or not specs:
        raise ValueError("eval.quality_gated_old_novelty.thresholds must be a non-empty list.")

    needs_support_quality = any(
        isinstance(spec, dict) and str(spec.get("mode", "fixed")) in {"support_quantile", "old_support_quantile"}
        for spec in specs
    )
    support_quality: np.ndarray | None = None
    if needs_support_quality:
        support_quality, _metadata = quality_scores_for_rows(
            support_rows,
            sample_rate=sample_rate,
            max_samples=max_samples,
            log_component="marginal_coverage_eval",
            log_label=f"fold_{fold_index}_support_threshold",
        )

    thresholds: list[tuple[str, float]] = []
    seen_names: set[str] = set()
    for spec in specs:
        name, value = _resolve_one_quality_gate_threshold(spec, support_quality=support_quality)
        if name in seen_names:
            raise ValueError(f"Duplicate quality gate threshold name: {name}")
        thresholds.append((name, value))
        seen_names.add(name)
    return thresholds


def _resolve_one_quality_gate_threshold(
    spec: object,
    *,
    support_quality: np.ndarray | None,
) -> tuple[str, float]:
    if isinstance(spec, int | float):
        value = float(spec)
        return _quality_threshold_name(value), _clip_quality_threshold(value)
    if not isinstance(spec, dict):
        raise ValueError("Quality gate threshold specs must be numbers or objects.")

    mode = str(spec.get("mode", "fixed"))
    if mode == "fixed":
        value = _clip_quality_threshold(float(spec.get("value", 0.45)))
        return str(spec.get("name", _quality_threshold_name(value))), value
    if mode in {"support_quantile", "old_support_quantile"}:
        if support_quality is None or len(support_quality) == 0:
            raise ValueError("Support-quantile quality gate requires non-empty support quality scores.")
        quantile = float(spec.get("quantile", 0.2))
        percentile = quantile * 100.0 if quantile <= 1.0 else quantile
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("Support quality quantile must be in [0, 1] or [0, 100].")
        value = _clip_quality_threshold(float(np.percentile(support_quality, percentile)))
        default_name = f"support_q{int(round(percentile)):02d}"
        return str(spec.get("name", default_name)), value
    raise ValueError(f"Unsupported quality gate threshold mode: {mode}")


def _quality_gate_source_caps(config: dict[str, Any]) -> list[int]:
    gate_config = config.get("eval", {}).get("quality_gated_old_novelty", {})
    if not isinstance(gate_config, dict) or not bool(gate_config.get("enabled", False)):
        return []
    source_caps = gate_config.get("source_caps", [])
    if source_caps is None:
        return []
    if not isinstance(source_caps, list):
        raise ValueError("eval.quality_gated_old_novelty.source_caps must be a list.")
    caps: list[int] = []
    for value in source_caps:
        if value is None:
            continue
        cap = int(value)
        if cap <= 0:
            raise ValueError("Quality-gated old-novelty source caps must be positive.")
        caps.append(cap)
    return caps


def _quality_gate_source_cap_key(config: dict[str, Any]) -> str:
    gate_config = config.get("eval", {}).get("quality_gated_old_novelty", {})
    source_key = str(gate_config.get("source_cap_key", "source_group_id"))
    if source_key not in {"source_group_id", "new_cluster_id", "new_cluster_parent_id"}:
        raise ValueError(
            "eval.quality_gated_old_novelty.source_cap_key must be "
            "'source_group_id', 'new_cluster_id', or 'new_cluster_parent_id'."
        )
    return source_key


def _source_cap_suffix(source_cap: int, source_key: str) -> str:
    if source_key == "source_group_id":
        return f"sourcecap{int(source_cap)}"
    if source_key == "new_cluster_id":
        return f"clustercap{int(source_cap)}"
    if source_key == "new_cluster_parent_id":
        return f"parentcap{int(source_cap)}"
    return f"{source_key.replace('_', '')}cap{int(source_cap)}"


def _quality_gate_validity_specs(config: dict[str, Any]) -> list[dict[str, object]]:
    gate_config = config.get("eval", {}).get("quality_gated_old_novelty", {})
    if not isinstance(gate_config, dict) or not bool(gate_config.get("enabled", False)):
        return []
    specs = gate_config.get("validity_gates", [])
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("eval.quality_gated_old_novelty.validity_gates must be a list.")
    output: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Quality-gated validity gate specs must be objects.")
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ValueError("Quality-gated validity gate specs require a non-empty name.")
        if name in seen_names:
            raise ValueError(f"Duplicate quality-gated validity gate name: {name}")
        if "max_stationary_fraction" not in spec and "max_abs_value" not in spec:
            raise ValueError(
                "Quality-gated validity gate specs require max_stationary_fraction or max_abs_value."
            )
        output_spec: dict[str, object] = {"name": name}
        if "max_stationary_fraction" in spec:
            max_stationary = float(spec["max_stationary_fraction"])
            if not 0.0 <= max_stationary <= 1.0:
                raise ValueError("max_stationary_fraction must be in [0, 1].")
            output_spec["max_stationary_fraction"] = max_stationary
        if "max_abs_value" in spec:
            max_abs_value = float(spec["max_abs_value"])
            if max_abs_value < 0.0:
                raise ValueError("max_abs_value must be non-negative.")
            output_spec["max_abs_value"] = max_abs_value
        output.append(output_spec)
        seen_names.add(name)
    return output


def _quality_gated_old_novelty_order(
    rows: list[dict[str, object]],
    *,
    quality_threshold: float,
    max_stationary_fraction: object | None = None,
    max_abs_value: object | None = None,
    source_cap: int | None = None,
    source_key: str = "source_group_id",
) -> list[int]:
    ranked = quality_gated_old_novelty_rank_rows(
        rows,
        quality_threshold=quality_threshold,
        max_stationary_fraction=(
            float(max_stationary_fraction) if max_stationary_fraction is not None else None
        ),
        max_abs_value=(float(max_abs_value) if max_abs_value is not None else None),
        source_cap=source_cap,
        source_key=source_key,
    )
    return [int(row["candidate_index"]) for row in ranked]


def _quality_gated_random_control_specs(config: dict[str, Any]) -> list[dict[str, object]]:
    return _control_policy_specs(config, key="quality_gated_random_controls")


def _kcenter_greedy_control_specs(config: dict[str, Any]) -> list[dict[str, object]]:
    return _control_policy_specs(config, key="kcenter_greedy_controls")


def _control_policy_specs(config: dict[str, Any], *, key: str) -> list[dict[str, object]]:
    specs = config.get("eval", {}).get(key, [])
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError(f"eval.{key} must be a list.")
    output: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError(f"eval.{key} entries must be objects.")
        name = _control_policy_name(spec, default=key.rstrip("s"))
        if name in seen_names:
            raise ValueError(f"Duplicate control policy name: {name}")
        output.append(dict(spec))
        seen_names.add(name)
    return output


def _control_policy_name(spec: dict[str, object], *, default: str) -> str:
    name = str(spec.get("name", default)).strip()
    if not name:
        raise ValueError("Control policy specs require a non-empty name.")
    return name


def _control_quality_threshold(spec: dict[str, object], *, default: float) -> float:
    return _clip_quality_threshold(float(spec.get("quality_threshold", default)))


def _optional_control_float(spec: dict[str, object], key: str) -> float | None:
    if key not in spec or spec[key] is None:
        return None
    return float(spec[key])


def _optional_control_int(spec: dict[str, object], key: str) -> int | None:
    if key not in spec or spec[key] is None:
        return None
    value = int(spec[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive when provided.")
    return value


def _passes_control_gates(
    row: dict[str, object],
    *,
    quality_threshold: float,
    max_stationary_fraction: float | None,
    max_abs_value: float | None,
) -> bool:
    if _safe_float(row.get("quality_score", 0.0)) < float(quality_threshold):
        return False
    if max_stationary_fraction is not None:
        if _safe_float(row.get("stationary_fraction", 0.0)) > float(max_stationary_fraction):
            return False
    if max_abs_value is not None:
        if _safe_float(row.get("max_abs_value", row.get("max_abs", 0.0))) > float(max_abs_value):
            return False
    return True


def _source_capped_order(
    order: Sequence[int],
    rows: list[dict[str, object]],
    *,
    source_cap: int,
    source_key: str,
) -> list[int]:
    selected: list[int] = []
    skipped: list[int] = []
    source_counts: defaultdict[str, int] = defaultdict(int)
    for idx in order:
        row = rows[int(idx)]
        if source_key not in row:
            raise ValueError(f"source_key {source_key} is missing from a random-control row.")
        source = str(row[source_key])
        if source_counts[source] < source_cap:
            selected.append(int(idx))
            source_counts[source] += 1
        else:
            skipped.append(int(idx))
    return [*selected, *skipped]


def _kcenter_greedy_quality_gated_order(
    *,
    scored: list[dict[str, object]],
    candidate_indices: np.ndarray,
    support_indices: np.ndarray,
    embeddings: np.ndarray,
    quality_threshold: float,
    max_stationary_fraction: float | None,
    max_abs_value: float | None,
) -> list[int]:
    eligible = [
        idx
        for idx, row in enumerate(scored)
        if _passes_control_gates(
            row,
            quality_threshold=quality_threshold,
            max_stationary_fraction=max_stationary_fraction,
            max_abs_value=max_abs_value,
        )
    ]
    eligible_order = _farthest_first_order(
        scored=scored,
        eligible_indices=eligible,
        candidate_embeddings=embeddings[np.asarray(candidate_indices, dtype=int)],
        support_embeddings=embeddings[np.asarray(support_indices, dtype=int)],
    )
    eligible_set = set(eligible_order)
    fallback = _quality_then_novelty_order(scored, [idx for idx in range(len(scored)) if idx not in eligible_set])
    return [*eligible_order, *fallback]


def _farthest_first_order(
    *,
    scored: list[dict[str, object]],
    eligible_indices: Sequence[int],
    candidate_embeddings: np.ndarray,
    support_embeddings: np.ndarray,
) -> list[int]:
    if not eligible_indices:
        return []
    candidates = normalize_rows(np.asarray(candidate_embeddings, dtype=float))
    support = normalize_rows(np.asarray(support_embeddings, dtype=float))
    nearest_similarity = np.max(candidates @ support.T, axis=1) if len(support) else np.full(len(candidates), -1.0)
    remaining = set(int(idx) for idx in eligible_indices)
    ordered: list[int] = []
    while remaining:
        best = sorted(
            remaining,
            key=lambda idx: (
                -(1.0 - float(nearest_similarity[int(idx)])),
                -_safe_float(scored[int(idx)].get("quality_score", 0.0)),
                str(scored[int(idx)].get("sample_id", "")),
            ),
        )[0]
        ordered.append(best)
        remaining.remove(best)
        selected_similarity = candidates @ candidates[best]
        nearest_similarity = np.maximum(nearest_similarity, selected_similarity)
    return ordered


def _quality_then_novelty_order(rows: list[dict[str, object]], indices: Sequence[int]) -> list[int]:
    return [
        int(idx)
        for idx in sorted(
            indices,
            key=lambda idx: (
                -_safe_float(rows[int(idx)].get("quality_score", 0.0)),
                -_safe_float(rows[int(idx)].get("old_novelty_score", 0.0)),
                str(rows[int(idx)].get("sample_id", "")),
            ),
        )
    ]


def _eval_rep_old_novelty_rows(
    *,
    scored: list[dict[str, object]],
    candidate_indices: np.ndarray,
    support_indices: np.ndarray,
    eval_embeddings: dict[str, np.ndarray],
    representation: str,
    k_old: int,
    cluster_similarity_threshold: float,
) -> list[int]:
    embeddings = eval_embeddings[representation]
    candidate_index_array = np.asarray(candidate_indices, dtype=int)
    support_index_array = np.asarray(support_indices, dtype=int)
    distances, _neighbors = old_knn_novelty(
        embeddings[support_index_array],
        embeddings[candidate_index_array],
        k=max(1, int(k_old)),
    )
    order_rows = []
    for local_index, row in enumerate(scored):
        order_rows.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "sample_id": str(row.get("sample_id", "")),
                "source_group_id": str(row.get("source_group_id", "")),
                "quality_score": _safe_float(row.get("quality_score", 0.0)),
                "stationary_fraction": _safe_float(row.get("stationary_fraction", 0.0)),
                "max_abs_value": _safe_float(row.get("max_abs_value", 0.0)),
                "extreme_value_fraction": _safe_float(row.get("extreme_value_fraction", 0.0)),
                "old_novelty_score": float(distances[local_index]),
                "final_score": float(distances[local_index]),
            }
        )
    cluster_ids = compute_batch_clusters(
        embeddings[candidate_index_array],
        similarity_threshold=float(cluster_similarity_threshold),
    )
    return annotate_cluster_features(order_rows, embeddings[candidate_index_array], cluster_ids)


def _eval_rep_old_novelty_order(rows: list[dict[str, object]]) -> list[int]:
    return [
        int(row["candidate_index"])
        for row in sorted(
            rows,
            key=lambda row: (
                -_safe_float(row["old_novelty_score"]),
                -_safe_float(row["quality_score"]),
                str(row["sample_id"]),
            ),
        )
    ]


def _quality_threshold_name(value: float) -> str:
    return f"q{int(round(_clip_quality_threshold(value) * 100)):02d}"


def _clip_quality_threshold(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _order_by_score(rows: list[dict[str, object]], columns: Sequence[str]) -> list[int]:
    score_column = str(columns[0])
    tie_column = str(columns[1]) if len(columns) > 1 else "sample_id"
    return [
        int(row["candidate_index"])
        for row in sorted(
            rows,
            key=lambda row: (-_safe_float(row.get(score_column, 0.0)), str(row.get(tie_column, ""))),
        )
    ]


def _diverse_source_cluster_order(rows: list[dict[str, object]]) -> list[int]:
    groups: defaultdict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("source_group_id", "")), int(row.get("new_cluster_id", -1)))
        groups[key].append(row)
    queues = [
        sorted(values, key=lambda row: (-_safe_float(row.get("final_score", 0.0)), str(row.get("sample_id", ""))))
        for _key, values in sorted(groups.items(), key=lambda item: str(item[0]))
    ]
    order: list[int] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(int(queue.pop(0)["candidate_index"]))
    return order


def _selection_summary(scored: list[dict[str, object]], selected: Sequence[int]) -> dict[str, object]:
    rows = [scored[int(idx)] for idx in selected]
    if not rows:
        return {
            "n_rows": 0,
            "mean_quality": 0.0,
            "mean_stationary_fraction": 0.0,
            "max_stationary_fraction": 0.0,
            "stationary_fraction_over_90": 0.0,
            "mean_max_abs_value": 0.0,
            "max_abs_value": 0.0,
            "max_abs_value_over_60": 0.0,
            "target_source_fraction": 0.0,
            "unique_source_groups": 0,
            "largest_source_group_fraction": 0.0,
        }
    source_groups = [str(row.get("source_group_id", "")) for row in rows]
    qualities = [_safe_float(row.get("quality_score", 0.0)) for row in rows]
    stationary = [_safe_float(row.get("stationary_fraction", 0.0)) for row in rows]
    max_abs = [_safe_float(row.get("max_abs_value", 0.0)) for row in rows]
    target_source = [int(row.get("is_target_source_group", 0)) for row in rows]
    return {
        "n_rows": int(len(rows)),
        "mean_quality": float(np.mean(qualities)),
        "min_quality": float(np.min(qualities)),
        "mean_stationary_fraction": float(np.mean(stationary)),
        "max_stationary_fraction": float(np.max(stationary)),
        "stationary_fraction_over_90": float(np.mean([value > 0.90 for value in stationary])),
        "mean_max_abs_value": float(np.mean(max_abs)),
        "max_abs_value": float(np.max(max_abs)),
        "max_abs_value_over_60": float(np.mean([value > 60.0 for value in max_abs])),
        "target_source_fraction": float(np.mean(target_source)),
        "unique_source_groups": int(len(set(source_groups))),
        "largest_source_group_fraction": _largest_fraction(source_groups),
        "candidate_type_counts": dict(Counter(str(row.get("candidate_type", "")) for row in rows)),
    }


def _mean_selection_summary(scored: list[dict[str, object]], selected_by_seed: Sequence[Sequence[int]]) -> dict[str, object]:
    summaries = [_selection_summary(scored, selected) for selected in selected_by_seed]
    if not summaries:
        return {}
    numeric_keys = [
        "n_rows",
        "mean_quality",
        "min_quality",
        "mean_stationary_fraction",
        "max_stationary_fraction",
        "stationary_fraction_over_90",
        "mean_max_abs_value",
        "max_abs_value",
        "max_abs_value_over_60",
        "target_source_fraction",
        "unique_source_groups",
        "largest_source_group_fraction",
    ]
    return {
        f"{key}_mean": float(np.mean([float(summary.get(key, 0.0)) for summary in summaries]))
        for key in numeric_keys
    }


def _mean_std_metrics(metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(metrics[0])
    output: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(metric.get(key, 0.0)) for metric in metrics], dtype=float)
        output[f"{key}_mean"] = float(np.mean(values))
        output[f"{key}_std"] = float(np.std(values))
    return output


def _mean_coverage_report(fold_reports: list[dict[str, Any]]) -> dict[str, float]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for fold in fold_reports:
        policies = fold.get("policies", {})
        if not isinstance(policies, dict):
            continue
        for policy_name, policy_report in policies.items():
            if not isinstance(policy_report, dict):
                continue
            for key, coverage_by_rep in policy_report.items():
                if not str(key).startswith("coverage@") or not isinstance(coverage_by_rep, dict):
                    continue
                for rep_name, metrics in coverage_by_rep.items():
                    if not isinstance(metrics, dict):
                        continue
                    metric_key = "relative_coverage_gain_mean" if "relative_coverage_gain_mean" in metrics else "relative_coverage_gain"
                    if metric_key in metrics:
                        values[f"{policy_name}.{key}.{rep_name}.{metric_key}"].append(float(metrics[metric_key]))
    return {key: float(np.mean(metric_values)) for key, metric_values in sorted(values.items())}


def _finite_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array.")
    if array.shape[0] == 0:
        raise ValueError("Expected at least one row.")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _source_group_id(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in PurePosixPath(parsed.path).parts if part and part != "/"]
    for part in parts:
        if part.startswith("worker"):
            return part
    return parsed.netloc + ":" + PurePosixPath(parsed.path).name


def _sample_rows(rows: list[SplitSample], *, max_rows: int, rng: np.random.Generator) -> list[SplitSample]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    indices = rng.choice(len(rows), size=max_rows, replace=False)
    return [rows[int(idx)] for idx in np.sort(indices)]


def _sample_indices(indices: np.ndarray, *, max_size: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if max_size <= 0 or len(values) <= max_size:
        return values
    sampled = rng.choice(values, size=max_size, replace=False)
    return np.asarray(sampled, dtype=int)


def _max_raw_samples(config: dict[str, Any]) -> int | None:
    eval_value = config.get("eval", {}).get("raw_shape_max_samples")
    if eval_value is not None:
        return int(eval_value)
    quality_value = config.get("quality", {}).get("max_samples_per_clip")
    return int(quality_value) if quality_value is not None else None


def _largest_fraction(values: list[str]) -> float:
    if not values:
        return 0.0
    return float(Counter(values).most_common(1)[0][1] / len(values))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_safe_row(row: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            safe[key] = value.item()
        elif isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe
