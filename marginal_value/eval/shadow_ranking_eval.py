from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from marginal_value.eval.ranking_audit import audit_ranking_artifacts
from marginal_value.logging_utils import log_event


DEFAULT_TOP_KS = [10, 50, 100]
DEFAULT_LOW_QUALITY_THRESHOLD = 0.45
DEFAULT_CLUSTER_BONUS_WEIGHT = 0.25
DEFAULT_DIVERSITY_METHOD = "cluster_bonus"
DEFAULT_CLUSTER_CAP_TOP_K = 100
DEFAULT_CLUSTER_MAX_PER_CLUSTER = 5
SHADOW_SCORE_VARIANT = "quality_gated_grammar"
SUPPORTED_DIVERSITY_METHODS = {
    "score_sort",
    "cluster_bonus",
    "cluster_cap",
    "cluster_mmr",
    "cluster_cap_then_cluster_mmr",
    "cluster_round_robin",
}
FORBIDDEN_SCORE_FEATURES = {
    "candidate_index",
    "cluster_bonus",
    "cluster_round",
    "cluster_selection_score",
    "duplicate_group_id",
    "fold_index",
    "heldout_cluster",
    "heldout_phrase_id",
    "is_artifact",
    "is_redundant",
    "label",
    "negative_type",
    "rank",
    "reason_code",
    "rerank_score",
    "reranker",
    "sample_id",
    "source_cluster",
    "split",
    "worker_id",
}


def run_shadow_ranking_eval(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    allow_local_execution: bool = False,
) -> dict[str, Any]:
    validate_shadow_ranking_config(config, allow_local_execution=allow_local_execution)
    artifacts = config["artifacts"]
    output = Path(output_dir) if output_dir is not None else Path(artifacts["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    diagnostics_path = Path(artifacts["diagnostics_path"])
    candidate_path = Path(artifacts["candidate_path"])
    quality_path = Path(artifacts["quality_metadata_path"])
    shadow_config = config.get("shadow", {})
    audit_config = config.get("audit", {})
    variant_specs = _shadow_variant_specs(shadow_config)
    top_ks = [int(value) for value in audit_config.get("top_ks", DEFAULT_TOP_KS)]
    low_quality_threshold = float(audit_config.get("low_quality_threshold", DEFAULT_LOW_QUALITY_THRESHOLD))
    n_examples = int(audit_config.get("n_examples", 25))

    log_event(
        "shadow_ranking",
        "start",
        diagnostics_path=str(diagnostics_path),
        candidate_path=str(candidate_path),
        output_dir=str(output),
    )
    diagnostics_rows = _read_csv(diagnostics_path)
    candidate_rows = _read_csv(candidate_path)
    quality_rows = _read_csv(quality_path)
    log_event(
        "shadow_ranking",
        "rows_read_done",
        n_diagnostics=len(diagnostics_rows),
        n_candidates=len(candidate_rows),
        n_quality=len(quality_rows),
    )

    source_artifacts = {
        "diagnostics_path": str(diagnostics_path),
        "candidate_path": str(candidate_path),
        "quality_metadata_path": str(quality_path),
    }
    if len(variant_specs) == 1:
        result = _run_shadow_variant(
            variant_specs[0],
            diagnostics_rows=diagnostics_rows,
            candidate_rows=candidate_rows,
            quality_rows=quality_rows,
            output=output,
            source_artifacts=source_artifacts,
            top_ks=top_ks,
            low_quality_threshold=low_quality_threshold,
            n_examples=n_examples,
        )
    else:
        variants: dict[str, Any] = {}
        summary_rows: list[dict[str, object]] = []
        for spec in variant_specs:
            variant_name = str(spec["name"])
            variant_result = _run_shadow_variant(
                spec,
                diagnostics_rows=diagnostics_rows,
                candidate_rows=candidate_rows,
                quality_rows=quality_rows,
                output=output / variant_name,
                source_artifacts=source_artifacts,
                top_ks=top_ks,
                low_quality_threshold=low_quality_threshold,
                n_examples=n_examples,
            )
            report = json.loads(Path(variant_result["report_path"]).read_text(encoding="utf-8"))
            variants[variant_name] = report
            summary_rows.append(_variant_summary_row(variant_name, report))
        report_path = output / "shadow_ranking_report_full.json"
        summary_path = output / "shadow_variant_summary_full.csv"
        selection = _select_variant(summary_rows, variants, config.get("selection_criteria", {}))
        report = {
            "mode": "full",
            "stage": "shadow_quality_gated_grammar_diversity_comparison",
            "score_variant": SHADOW_SCORE_VARIANT,
            "selected_variant": selection["selected_variant"],
            "selection_criteria": selection,
            "source_artifacts": source_artifacts,
            "output_artifacts": {
                "report_path": str(report_path),
                "summary_path": str(summary_path),
            },
            "variants": variants,
            "variant_summary": summary_rows,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        _write_rows(summary_path, summary_rows)
        best = selection["selected_summary"] if selection["selected_summary"] else _best_variant_summary(summary_rows)
        result = {
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "variant_names": [str(spec["name"]) for spec in variant_specs],
            "best_variant": best.get("variant", ""),
            "shadow_top100_positive_fraction": best.get("candidate_top100_positive_fraction", 0.0),
            "shadow_top100_unique_cluster_count": best.get("candidate_top100_unique_cluster_count", 0),
        }
    log_event("shadow_ranking", "done", **result)
    return result


def _run_shadow_variant(
    variant_spec: dict[str, Any],
    *,
    diagnostics_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
    quality_rows: Sequence[dict[str, object]],
    output: Path,
    source_artifacts: dict[str, str],
    top_ks: Sequence[int],
    low_quality_threshold: float,
    n_examples: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    ranked_diagnostics, diagnostics_metadata = build_shadow_ranked_rows(
        diagnostics_rows,
        diversity_method=str(variant_spec["diversity_method"]),
        cluster_bonus_weight=float(variant_spec.get("cluster_bonus_weight", DEFAULT_CLUSTER_BONUS_WEIGHT)),
        cluster_cap_top_k=int(variant_spec.get("cluster_cap_top_k", DEFAULT_CLUSTER_CAP_TOP_K)),
        cluster_max_per_cluster=int(variant_spec.get("cluster_max_per_cluster", DEFAULT_CLUSTER_MAX_PER_CLUSTER)),
        lambda_redundancy=float(variant_spec.get("lambda_redundancy", 0.0)),
    )
    ranked_candidates, candidate_metadata = build_shadow_ranked_rows(
        candidate_rows,
        diversity_method=str(variant_spec["diversity_method"]),
        cluster_bonus_weight=float(variant_spec.get("cluster_bonus_weight", DEFAULT_CLUSTER_BONUS_WEIGHT)),
        cluster_cap_top_k=int(variant_spec.get("cluster_cap_top_k", DEFAULT_CLUSTER_CAP_TOP_K)),
        cluster_max_per_cluster=int(variant_spec.get("cluster_max_per_cluster", DEFAULT_CLUSTER_MAX_PER_CLUSTER)),
        lambda_redundancy=float(variant_spec.get("lambda_redundancy", 0.0)),
    )
    comparison = compare_current_and_shadow_candidates(
        candidate_rows,
        ranked_candidates,
        top_ks=top_ks,
        low_quality_threshold=low_quality_threshold,
    )

    submission_path = output / "shadow_submission_val_full.csv"
    diagnostics_out_path = output / "shadow_diagnostics_val_full.csv"
    candidate_out_path = output / "shadow_ranking_val_candidates_full.csv"
    quality_out_path = output / "shadow_quality_metadata_full.csv"
    report_path = output / "shadow_ranking_report_full.json"
    top_examples_path = output / "shadow_top_examples_full.csv"
    top_movers_path = output / "shadow_top_movers_full.csv"

    _write_rows(submission_path, _submission_rows(ranked_diagnostics))
    _write_rows(diagnostics_out_path, ranked_diagnostics)
    _write_rows(candidate_out_path, ranked_candidates)
    _write_rows(quality_out_path, quality_rows)

    audit = audit_ranking_artifacts(
        submission_path=submission_path,
        diagnostics_path=diagnostics_out_path,
        candidate_path=candidate_out_path,
        quality_metadata_path=quality_out_path,
        top_ks=top_ks,
        low_quality_threshold=low_quality_threshold,
        n_examples=n_examples,
    )
    report = {
        "mode": "full",
        "stage": "shadow_quality_gated_grammar_ranking",
        "shadow": {
            "name": variant_spec["name"],
            "score_variant": SHADOW_SCORE_VARIANT,
            "diversity_method": variant_spec["diversity_method"],
            "cluster_bonus_weight": float(variant_spec.get("cluster_bonus_weight", DEFAULT_CLUSTER_BONUS_WEIGHT)),
            "cluster_cap_top_k": int(variant_spec.get("cluster_cap_top_k", DEFAULT_CLUSTER_CAP_TOP_K)),
            "cluster_max_per_cluster": int(variant_spec.get("cluster_max_per_cluster", DEFAULT_CLUSTER_MAX_PER_CLUSTER)),
            "lambda_redundancy": float(variant_spec.get("lambda_redundancy", 0.0)),
            "diagnostics_metadata": diagnostics_metadata,
            "candidate_metadata": candidate_metadata,
        },
        "source_artifacts": source_artifacts,
        "output_artifacts": {
            "submission_path": str(submission_path),
            "diagnostics_path": str(diagnostics_out_path),
            "candidate_path": str(candidate_out_path),
            "quality_metadata_path": str(quality_out_path),
            "top_examples_path": str(top_examples_path),
            "top_movers_path": str(top_movers_path),
            "report_path": str(report_path),
        },
        "candidate_comparison": comparison,
        "audit": audit,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows(top_examples_path, ranked_diagnostics[:n_examples])
    _write_rows(top_movers_path, _top_candidate_movers(candidate_rows, ranked_candidates, n_examples=n_examples))

    return {
        "report_path": str(report_path),
        "submission_path": str(submission_path),
        "diagnostics_path": str(diagnostics_out_path),
        "candidate_path": str(candidate_out_path),
        "top_examples_path": str(top_examples_path),
        "top_movers_path": str(top_movers_path),
        "shadow_top100_positive_fraction": comparison.get("shadow", {}).get("top_k", {}).get("100", {}).get("positive_fraction", 0.0),
        "shadow_top100_unique_cluster_count": comparison.get("shadow", {}).get("top_k", {}).get("100", {}).get("unique_cluster_count", 0),
    }


def build_shadow_ranked_rows(
    rows: Sequence[dict[str, object]],
    *,
    diversity_method: str = DEFAULT_DIVERSITY_METHOD,
    cluster_bonus_weight: float = DEFAULT_CLUSTER_BONUS_WEIGHT,
    cluster_cap_top_k: int = DEFAULT_CLUSTER_CAP_TOP_K,
    cluster_max_per_cluster: int = DEFAULT_CLUSTER_MAX_PER_CLUSTER,
    lambda_redundancy: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    if not rows:
        return [], {"used_features": [], "forbidden_used": []}
    if diversity_method not in SUPPORTED_DIVERSITY_METHODS:
        raise ValueError(f"Unsupported diversity_method '{diversity_method}'.")
    scores, used_features = _quality_gated_grammar_scores(rows)
    forbidden_used = sorted(set(used_features) & FORBIDDEN_SCORE_FEATURES)
    if forbidden_used:
        raise ValueError(f"Shadow score uses forbidden features: {forbidden_used}")
    cluster_ids = [str(row.get("new_cluster_id", idx)) for idx, row in enumerate(rows)]
    order, selection_scores, cluster_rounds, cluster_bonuses = _diversity_aware_order(
        scores,
        cluster_ids,
        diversity_method=diversity_method,
        cluster_bonus_weight=cluster_bonus_weight,
        cluster_cap_top_k=cluster_cap_top_k,
        cluster_max_per_cluster=cluster_max_per_cluster,
        lambda_redundancy=lambda_redundancy,
    )
    ranked: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        source = dict(rows[idx])
        original_rank = source.get("rank", "")
        original_score = source.get("final_score", "")
        original_reason = source.get("reason_code", "")
        source["original_rank"] = original_rank
        source["original_final_score"] = original_score
        source["original_reason_code"] = original_reason
        source["rank"] = rank
        source["shadow_score"] = float(scores[idx])
        source["shadow_selection_score"] = float(selection_scores[idx])
        source["shadow_rerank_score"] = _monotone_rank_score(rank, len(order), float(scores[idx]))
        source["shadow_cluster_bonus"] = float(cluster_bonuses[idx])
        source["shadow_cluster_round"] = int(cluster_rounds[idx])
        source["shadow_score_variant"] = SHADOW_SCORE_VARIANT
        source["shadow_diversity_method"] = diversity_method
        source["final_score"] = float(scores[idx])
        source["reason_code"] = _shadow_reason_code(source)
        ranked.append(source)
    metadata = {
        "score_variant": SHADOW_SCORE_VARIANT,
        "diversity_method": diversity_method,
        "used_features": used_features,
        "forbidden_used": forbidden_used,
        "cluster_bonus_weight": cluster_bonus_weight,
        "cluster_cap_top_k": cluster_cap_top_k,
        "cluster_max_per_cluster": cluster_max_per_cluster,
        "lambda_redundancy": lambda_redundancy,
        "n_rows": len(rows),
    }
    return ranked, metadata


def compare_current_and_shadow_candidates(
    current_rows: Sequence[dict[str, object]],
    shadow_rows: Sequence[dict[str, object]],
    *,
    top_ks: Iterable[int],
    low_quality_threshold: float,
) -> dict[str, Any]:
    current_ranked = sorted((dict(row) for row in current_rows), key=lambda row: _to_float(row.get("rank", 0), default=0.0))
    shadow_ranked = sorted((dict(row) for row in shadow_rows), key=lambda row: _to_float(row.get("rank", 0), default=0.0))
    return {
        "current": _candidate_rank_summary(current_ranked, top_ks=top_ks, low_quality_threshold=low_quality_threshold),
        "shadow": _candidate_rank_summary(shadow_ranked, top_ks=top_ks, low_quality_threshold=low_quality_threshold),
        "grammar_feature_coverage_by_label": _grammar_feature_coverage_by_label(current_ranked),
        "candidate_eval_score_leakage_risk": _candidate_eval_score_leakage_risk(current_ranked),
        "rank_delta": _rank_delta_summary(current_ranked, shadow_ranked),
    }


def validate_shadow_ranking_config(config: dict[str, Any], *, allow_local_execution: bool = False) -> None:
    execution = _required_mapping(config, "execution")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal" and not allow_local_execution:
        raise ValueError("Shadow ranking must be dispatched through Modal.")
    for key in ("diagnostics_path", "candidate_path", "quality_metadata_path", "output_dir"):
        value = Path(str(artifacts.get(key, "")))
        if not value.is_absolute():
            raise ValueError(f"artifacts.{key} must be absolute.")
    score_variant = str(config.get("shadow", {}).get("score_variant", SHADOW_SCORE_VARIANT))
    if score_variant != SHADOW_SCORE_VARIANT:
        raise ValueError(f"shadow.score_variant must be '{SHADOW_SCORE_VARIANT}'.")
    for spec in _shadow_variant_specs(config.get("shadow", {})):
        cluster_bonus_weight = float(spec.get("cluster_bonus_weight", DEFAULT_CLUSTER_BONUS_WEIGHT))
        if not 0.0 <= cluster_bonus_weight <= 1.0:
            raise ValueError("shadow.cluster_bonus_weight must be in [0, 1].")
        if spec["diversity_method"] not in SUPPORTED_DIVERSITY_METHODS:
            raise ValueError(f"Unsupported diversity_method '{spec['diversity_method']}'.")
        if int(spec.get("cluster_cap_top_k", DEFAULT_CLUSTER_CAP_TOP_K)) < 1:
            raise ValueError("shadow.cluster_cap_top_k must be positive.")
        if int(spec.get("cluster_max_per_cluster", DEFAULT_CLUSTER_MAX_PER_CLUSTER)) < 1:
            raise ValueError("shadow.cluster_max_per_cluster must be positive.")
        lambda_redundancy = float(spec.get("lambda_redundancy", 0.0))
        if not 0.0 <= lambda_redundancy <= 1.0:
            raise ValueError("shadow.lambda_redundancy must be in [0, 1].")


def load_shadow_ranking_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _shadow_variant_specs(shadow_config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_variants = shadow_config.get("diversity_variants")
    if raw_variants is None:
        raw_variants = [
            {
                "name": shadow_config.get("name", shadow_config.get("diversity_method", DEFAULT_DIVERSITY_METHOD)),
                "diversity_method": shadow_config.get("diversity_method", DEFAULT_DIVERSITY_METHOD),
                "cluster_bonus_weight": shadow_config.get("cluster_bonus_weight", DEFAULT_CLUSTER_BONUS_WEIGHT),
                "cluster_cap_top_k": shadow_config.get("cluster_cap_top_k", DEFAULT_CLUSTER_CAP_TOP_K),
                "cluster_max_per_cluster": shadow_config.get("cluster_max_per_cluster", DEFAULT_CLUSTER_MAX_PER_CLUSTER),
                "lambda_redundancy": shadow_config.get("lambda_redundancy", 0.0),
            }
        ]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("shadow.diversity_variants must be a non-empty list when provided.")
    specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, raw in enumerate(raw_variants):
        if not isinstance(raw, dict):
            raise ValueError("Each shadow diversity variant must be an object.")
        method = str(raw.get("diversity_method", DEFAULT_DIVERSITY_METHOD))
        name = _sanitize_variant_name(str(raw.get("name", method)))
        if name in seen_names:
            name = f"{name}_{idx + 1}"
        seen_names.add(name)
        specs.append(
            {
                "name": name,
                "diversity_method": method,
                "cluster_bonus_weight": float(raw.get("cluster_bonus_weight", shadow_config.get("cluster_bonus_weight", DEFAULT_CLUSTER_BONUS_WEIGHT))),
                "cluster_cap_top_k": int(raw.get("cluster_cap_top_k", shadow_config.get("cluster_cap_top_k", DEFAULT_CLUSTER_CAP_TOP_K))),
                "cluster_max_per_cluster": int(raw.get("cluster_max_per_cluster", shadow_config.get("cluster_max_per_cluster", DEFAULT_CLUSTER_MAX_PER_CLUSTER))),
                "lambda_redundancy": float(raw.get("lambda_redundancy", shadow_config.get("lambda_redundancy", 0.0))),
            }
        )
    return specs


def _sanitize_variant_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return cleaned or DEFAULT_DIVERSITY_METHOD


def _quality_gated_grammar_scores(rows: Sequence[dict[str, object]]) -> tuple[np.ndarray, list[str]]:
    token_nll = _normalize(_column(rows, "token_nll_p95"))
    transition_nll = _normalize(_column(rows, "transition_nll_p95"))
    longest_unseen = _normalize(_column(rows, "longest_unseen_phrase_len"))
    quality = np.clip(_column(rows, "quality_score", default=1.0), 0.0, 1.0)
    grammar = _normalize(0.65 * token_nll + 0.25 * transition_nll + 0.10 * longest_unseen)
    return grammar * quality, ["token_nll_p95", "transition_nll_p95", "longest_unseen_phrase_len", "quality_score"]


def _diversity_aware_order(
    scores: np.ndarray,
    cluster_ids: Sequence[str],
    *,
    diversity_method: str,
    cluster_bonus_weight: float,
    cluster_cap_top_k: int,
    cluster_max_per_cluster: int,
    lambda_redundancy: float = 0.0,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    if diversity_method == "score_sort":
        return _cluster_bonus_order(scores, cluster_ids, cluster_bonus_weight=0.0)
    if diversity_method == "cluster_bonus":
        return _cluster_bonus_order(scores, cluster_ids, cluster_bonus_weight=cluster_bonus_weight)
    if diversity_method == "cluster_mmr":
        return _cluster_mmr_order(scores, cluster_ids, lambda_redundancy=lambda_redundancy or cluster_bonus_weight)
    if diversity_method == "cluster_cap":
        return _cluster_cap_order(
            scores,
            cluster_ids,
            cluster_bonus_weight=cluster_bonus_weight,
            cluster_cap_top_k=cluster_cap_top_k,
            cluster_max_per_cluster=cluster_max_per_cluster,
        )
    if diversity_method == "cluster_cap_then_cluster_mmr":
        return _cluster_cap_order(
            scores,
            cluster_ids,
            cluster_bonus_weight=0.0,
            cluster_cap_top_k=cluster_cap_top_k,
            cluster_max_per_cluster=cluster_max_per_cluster,
            lambda_redundancy=lambda_redundancy or cluster_bonus_weight,
        )
    if diversity_method == "cluster_round_robin":
        return _cluster_round_robin_order(scores, cluster_ids, cluster_bonus_weight=cluster_bonus_weight)
    raise ValueError(f"Unsupported diversity_method '{diversity_method}'.")


def _cluster_bonus_order(
    scores: np.ndarray,
    cluster_ids: Sequence[str],
    *,
    cluster_bonus_weight: float,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    remaining = np.ones(len(scores), dtype=bool)
    selected: list[int] = []
    seen_clusters: set[str] = set()
    selection_scores = np.zeros(len(scores), dtype=float)
    cluster_rounds = np.zeros(len(scores), dtype=int)
    cluster_bonuses = np.zeros(len(scores), dtype=float)
    cluster_selection_counts: Counter[str] = Counter()
    while len(selected) < len(scores):
        adjusted = np.asarray(scores, dtype=float).copy()
        bonuses = np.asarray([cluster_bonus_weight if cluster not in seen_clusters else 0.0 for cluster in cluster_ids], dtype=float)
        adjusted += bonuses
        adjusted[~remaining] = -np.inf
        best_idx = int(np.argmax(adjusted))
        if not np.isfinite(adjusted[best_idx]):
            break
        cluster = str(cluster_ids[best_idx])
        selection_scores[best_idx] = float(adjusted[best_idx])
        cluster_bonuses[best_idx] = float(bonuses[best_idx])
        cluster_rounds[best_idx] = int(cluster_selection_counts[cluster])
        selected.append(best_idx)
        remaining[best_idx] = False
        seen_clusters.add(cluster)
        cluster_selection_counts[cluster] += 1
    return selected, selection_scores, cluster_rounds, cluster_bonuses


def _cluster_cap_order(
    scores: np.ndarray,
    cluster_ids: Sequence[str],
    *,
    cluster_bonus_weight: float,
    cluster_cap_top_k: int,
    cluster_max_per_cluster: int,
    lambda_redundancy: float = 0.0,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    if cluster_cap_top_k < 1:
        raise ValueError("cluster_cap_top_k must be positive.")
    if cluster_max_per_cluster < 1:
        raise ValueError("cluster_max_per_cluster must be positive.")
    remaining = np.ones(len(scores), dtype=bool)
    selected: list[int] = []
    seen_clusters: set[str] = set()
    selection_scores = np.zeros(len(scores), dtype=float)
    cluster_rounds = np.zeros(len(scores), dtype=int)
    cluster_bonuses = np.zeros(len(scores), dtype=float)
    cluster_selection_counts: Counter[str] = Counter()
    while len(selected) < len(scores):
        under_cap = np.asarray(
            [
                cluster_selection_counts[str(cluster)] < cluster_max_per_cluster
                for cluster in cluster_ids
            ],
            dtype=bool,
        )
        eligible = remaining.copy()
        if len(selected) < cluster_cap_top_k and np.any(eligible & under_cap):
            eligible &= under_cap
        adjusted = _bonus_adjusted_scores(scores, cluster_ids, seen_clusters, cluster_bonus_weight)
        if lambda_redundancy > 0.0:
            penalties = np.asarray(
                [lambda_redundancy if cluster_selection_counts[str(cluster)] > 0 else 0.0 for cluster in cluster_ids],
                dtype=float,
            )
            adjusted -= penalties
        adjusted[~eligible] = -np.inf
        best_idx = int(np.argmax(adjusted))
        if not np.isfinite(adjusted[best_idx]):
            break
        cluster = str(cluster_ids[best_idx])
        selection_scores[best_idx] = float(adjusted[best_idx])
        cluster_bonuses[best_idx] = float(cluster_bonus_weight if cluster not in seen_clusters else 0.0)
        cluster_rounds[best_idx] = int(cluster_selection_counts[cluster])
        selected.append(best_idx)
        remaining[best_idx] = False
        seen_clusters.add(cluster)
        cluster_selection_counts[cluster] += 1
    return selected, selection_scores, cluster_rounds, cluster_bonuses


def _cluster_mmr_order(
    scores: np.ndarray,
    cluster_ids: Sequence[str],
    *,
    lambda_redundancy: float,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    remaining = np.ones(len(scores), dtype=bool)
    selected: list[int] = []
    selection_scores = np.zeros(len(scores), dtype=float)
    cluster_rounds = np.zeros(len(scores), dtype=int)
    cluster_bonuses = np.zeros(len(scores), dtype=float)
    cluster_selection_counts: Counter[str] = Counter()
    while len(selected) < len(scores):
        adjusted = np.asarray(scores, dtype=float).copy()
        penalties = np.asarray(
            [lambda_redundancy if cluster_selection_counts[str(cluster)] > 0 else 0.0 for cluster in cluster_ids],
            dtype=float,
        )
        adjusted -= penalties
        adjusted[~remaining] = -np.inf
        best_idx = int(np.argmax(adjusted))
        if not np.isfinite(adjusted[best_idx]):
            break
        cluster = str(cluster_ids[best_idx])
        selection_scores[best_idx] = float(adjusted[best_idx])
        cluster_rounds[best_idx] = int(cluster_selection_counts[cluster])
        selected.append(best_idx)
        remaining[best_idx] = False
        cluster_selection_counts[cluster] += 1
    return selected, selection_scores, cluster_rounds, cluster_bonuses


def _cluster_round_robin_order(
    scores: np.ndarray,
    cluster_ids: Sequence[str],
    *,
    cluster_bonus_weight: float,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    by_cluster: dict[str, list[int]] = {}
    for idx, cluster in enumerate(cluster_ids):
        by_cluster.setdefault(str(cluster), []).append(idx)
    for cluster, indices in by_cluster.items():
        by_cluster[cluster] = sorted(indices, key=lambda idx: (-float(scores[idx]), idx))
    cluster_order = sorted(
        by_cluster,
        key=lambda cluster: (-float(scores[by_cluster[cluster][0]]), cluster),
    )
    selected: list[int] = []
    selection_scores = np.zeros(len(scores), dtype=float)
    cluster_rounds = np.zeros(len(scores), dtype=int)
    cluster_bonuses = np.zeros(len(scores), dtype=float)
    seen_clusters: set[str] = set()
    round_index = 0
    while len(selected) < len(scores):
        selected_this_round = 0
        for cluster in cluster_order:
            indices = by_cluster[cluster]
            if round_index >= len(indices):
                continue
            idx = indices[round_index]
            bonus = cluster_bonus_weight if cluster not in seen_clusters else 0.0
            selection_scores[idx] = float(scores[idx]) + bonus
            cluster_bonuses[idx] = float(bonus)
            cluster_rounds[idx] = round_index
            selected.append(idx)
            seen_clusters.add(cluster)
            selected_this_round += 1
        if selected_this_round == 0:
            break
        round_index += 1
    return selected, selection_scores, cluster_rounds, cluster_bonuses


def _bonus_adjusted_scores(
    scores: np.ndarray,
    cluster_ids: Sequence[str],
    seen_clusters: set[str],
    cluster_bonus_weight: float,
) -> np.ndarray:
    bonuses = np.asarray([cluster_bonus_weight if cluster not in seen_clusters else 0.0 for cluster in cluster_ids], dtype=float)
    return np.asarray(scores, dtype=float).copy() + bonuses


def _monotone_rank_score(rank: int, n_rows: int, raw_score: float) -> float:
    if n_rows <= 1:
        return 1.0
    base = 1.0 - ((rank - 1) / n_rows)
    return float(base + 1.0e-9 * max(0.0, min(1.0, raw_score)))


def _variant_summary_row(variant_name: str, report: dict[str, Any]) -> dict[str, object]:
    candidate_top = report.get("candidate_comparison", {}).get("shadow", {}).get("top_k", {})
    candidate_key, candidate_metrics = _preferred_top_k(candidate_top)
    audit_top = report.get("audit", {}).get("top_k", {})
    audit_key, audit_metrics = _preferred_top_k(audit_top)
    shadow = report.get("shadow", {})
    return {
        "variant": variant_name,
        "diversity_method": shadow.get("diversity_method", ""),
        "cluster_bonus_weight": shadow.get("cluster_bonus_weight", 0.0),
        "cluster_cap_top_k": shadow.get("cluster_cap_top_k", ""),
        "cluster_max_per_cluster": shadow.get("cluster_max_per_cluster", ""),
        "lambda_redundancy": shadow.get("lambda_redundancy", ""),
        "candidate_eval_k": candidate_key,
        "candidate_top100_positive_fraction": candidate_metrics.get("positive_fraction", 0.0),
        "candidate_top100_unique_cluster_count": candidate_metrics.get("unique_cluster_count", 0),
        "candidate_top100_cluster_repeat_count": candidate_metrics.get("cluster_repeat_count", 0),
        "candidate_top100_low_quality_count": candidate_metrics.get("low_quality_count", 0),
        "diagnostics_eval_k": audit_key,
        "diagnostics_top100_unique_cluster_count": audit_metrics.get("unique_cluster_count", 0),
        "diagnostics_top100_low_quality_count": audit_metrics.get("low_quality_count", 0),
        "diagnostics_top100_mean_quality": audit_metrics.get("mean_quality", 0.0),
        "submission_score_nonincreasing": report.get("audit", {}).get("submission", {}).get("score_nonincreasing", False),
    }


def _best_variant_summary(summary_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not summary_rows:
        return {}
    return max(
        summary_rows,
        key=lambda row: (
            _to_float(row.get("candidate_top100_positive_fraction", 0.0), default=0.0),
            _to_float(row.get("candidate_top100_unique_cluster_count", 0), default=0.0),
            -_to_float(row.get("candidate_top100_low_quality_count", 0), default=0.0),
        ),
    )


def _select_variant(
    summary_rows: Sequence[dict[str, object]],
    variants: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    top_k = str(int(criteria.get("candidate_top_k", 100)))
    min_positive_fraction = float(criteria.get("min_positive_fraction", 0.0))
    min_unique_clusters = int(criteria.get("min_unique_clusters", 0))
    max_low_quality_count = int(criteria.get("max_low_quality_count", 10**9))
    candidates: list[tuple[dict[str, object], dict[str, Any]]] = []
    for row in summary_rows:
        name = str(row.get("variant", ""))
        metrics = variants.get(name, {}).get("candidate_comparison", {}).get("shadow", {}).get("top_k", {}).get(top_k, {})
        if not metrics:
            continue
        if float(metrics.get("positive_fraction", 0.0)) < min_positive_fraction:
            continue
        if int(metrics.get("unique_cluster_count", 0)) < min_unique_clusters:
            continue
        if int(metrics.get("low_quality_count", 0)) > max_low_quality_count:
            continue
        candidates.append((dict(row), metrics))
    if candidates:
        selected, metrics = min(
            candidates,
            key=lambda item: (
                float(item[1].get("dominant_cluster_fraction", 1.0)),
                -float(item[1].get("positive_fraction", 0.0)),
                -int(item[1].get("unique_cluster_count", 0)),
                str(item[0].get("variant", "")),
            ),
        )
        return {
            "candidate_top_k": int(top_k),
            "min_positive_fraction": min_positive_fraction,
            "min_unique_clusters": min_unique_clusters,
            "max_low_quality_count": max_low_quality_count,
            "passed": True,
            "selected_variant": selected.get("variant", ""),
            "selected_summary": selected,
            "selected_metrics": metrics,
        }
    fallback = _best_variant_summary(summary_rows)
    return {
        "candidate_top_k": int(top_k),
        "min_positive_fraction": min_positive_fraction,
        "min_unique_clusters": min_unique_clusters,
        "max_low_quality_count": max_low_quality_count,
        "passed": False,
        "selected_variant": fallback.get("variant", ""),
        "selected_summary": fallback,
        "selected_metrics": {},
    }


def _preferred_top_k(top_k: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not top_k:
        return "", {}
    if "100" in top_k:
        return "100", top_k["100"]
    key = max(top_k, key=lambda value: int(value))
    return key, top_k[key]


def _shadow_reason_code(row: dict[str, object]) -> str:
    quality = _to_float(row.get("quality_score", 1.0), default=1.0)
    score = _to_float(row.get("shadow_score", row.get("final_score", 0.0)), default=0.0)
    novelty = _to_float(row.get("old_novelty_score", 0.0), default=0.0)
    density = _to_float(row.get("new_density_score", row.get("new_batch_density", 0.0)), default=0.0)
    if quality < DEFAULT_LOW_QUALITY_THRESHOLD:
        return "LOW_QUALITY"
    if score >= 0.65:
        return "RARE_TEMPORAL_COMPOSITION"
    if novelty >= 0.65 and density >= 0.45:
        return "COHESIVE_NEW_WORKFLOW"
    if novelty < 0.35 and density >= 0.65:
        return "REDUNDANT_KNOWN_WORKFLOW"
    return "RARE_MOTION_PRIMITIVES"


def _submission_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "worker_id": row.get("worker_id", row.get("sample_id", "")),
            "rank": int(row["rank"]),
            "score": float(row.get("shadow_rerank_score", row.get("shadow_score", row.get("final_score", 0.0)))),
            "quality_score": float(_to_float(row.get("quality_score", 1.0), default=1.0)),
            "reason_code": row.get("reason_code", ""),
        }
        for row in rows
    ]


def _candidate_rank_summary(
    rows: Sequence[dict[str, object]],
    *,
    top_ks: Iterable[int],
    low_quality_threshold: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "positive_rate": _mean([_to_int(row.get("label", 0)) for row in rows]),
        "top_k": {},
    }
    for k in top_ks:
        subset = list(rows)[: int(k)]
        labels = [_to_int(row.get("label", 0)) for row in subset]
        quality = [_to_float(row.get("quality_score", 1.0), default=1.0) for row in subset]
        clusters = [str(row.get("new_cluster_id", idx)) for idx, row in enumerate(subset)]
        reasons = Counter(str(row.get("reason_code", "")) for row in subset)
        cluster_summary = _cluster_distribution_summary(clusters)
        summary["top_k"][str(k)] = {
            "n_rows": len(subset),
            "positive_count": int(sum(labels)),
            "positive_fraction": _mean(labels),
            "low_quality_count": int(sum(value < low_quality_threshold for value in quality)),
            "mean_quality": _mean(quality),
            "unique_cluster_count": len(set(clusters)),
            "cluster_repeat_count": max(0, len(subset) - len(set(clusters))),
            "dominant_cluster_id": cluster_summary["dominant_cluster_id"],
            "dominant_cluster_count": cluster_summary["dominant_cluster_count"],
            "dominant_cluster_fraction": cluster_summary["dominant_cluster_fraction"],
            "largest_cluster_id": cluster_summary["dominant_cluster_id"],
            "largest_cluster_count": cluster_summary["dominant_cluster_count"],
            "largest_cluster_fraction": cluster_summary["dominant_cluster_fraction"],
            "cluster_gini": cluster_summary["cluster_gini"],
            "mean_per_cluster_count": cluster_summary["mean_per_cluster_count"],
            "top_cluster_counts": cluster_summary["top_cluster_counts"],
            "corruption_negative_count": int(sum(_truthy(row.get("is_corruption", "")) for row in subset)),
            "corruption_negative_fraction": _mean([1.0 if _truthy(row.get("is_corruption", "")) else 0.0 for row in subset]),
            "reason_code_counts": {key: int(value) for key, value in sorted(reasons.items())},
        }
    return summary


def _rank_delta_summary(current_rows: Sequence[dict[str, object]], shadow_rows: Sequence[dict[str, object]]) -> dict[str, Any]:
    current_rank_by_id = {_row_id(row): int(_to_float(row.get("rank", idx + 1), default=idx + 1)) for idx, row in enumerate(current_rows)}
    deltas: list[float] = []
    for idx, row in enumerate(shadow_rows):
        row_id = _row_id(row)
        if row_id not in current_rank_by_id:
            continue
        shadow_rank = int(_to_float(row.get("rank", idx + 1), default=idx + 1))
        deltas.append(float(current_rank_by_id[row_id] - shadow_rank))
    return {
        "count": len(deltas),
        "mean_rank_improvement": _mean(deltas),
        "max_rank_improvement": max(deltas) if deltas else 0.0,
        "max_rank_drop": min(deltas) if deltas else 0.0,
    }


def _grammar_feature_coverage_by_label(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_label: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if "label" not in row:
            continue
        by_label.setdefault(str(_to_int(row.get("label", 0))), []).append(row)
    coverage: dict[str, dict[str, float]] = {}
    for label, label_rows in sorted(by_label.items()):
        present = [_truthy(row.get("grammar_feature_present", False)) for row in label_rows]
        nonzero = [
            abs(_to_float(row.get("token_nll_p95", 0.0), default=0.0)) > 1.0e-12
            or abs(_to_float(row.get("transition_nll_p95", 0.0), default=0.0)) > 1.0e-12
            for row in label_rows
        ]
        coverage[label] = {
            "n_rows": float(len(label_rows)),
            "grammar_feature_present_fraction": _mean([1.0 if value else 0.0 for value in present]),
            "nonzero_grammar_score_fraction": _mean([1.0 if value else 0.0 for value in nonzero]),
        }
    return coverage


def _candidate_eval_score_leakage_risk(rows: Sequence[dict[str, object]]) -> bool:
    coverage = _grammar_feature_coverage_by_label(rows)
    if len(coverage) < 2:
        return False
    present = [value["grammar_feature_present_fraction"] for value in coverage.values()]
    nonzero = [value["nonzero_grammar_score_fraction"] for value in coverage.values()]
    return bool((max(present) - min(present) > 0.05) or (max(nonzero) - min(nonzero) > 0.05))


def _cluster_distribution_summary(clusters: Sequence[str]) -> dict[str, Any]:
    counts = Counter(clusters)
    if not counts:
        return {
            "dominant_cluster_id": "",
            "dominant_cluster_count": 0,
            "dominant_cluster_fraction": 0.0,
            "cluster_gini": 0.0,
            "mean_per_cluster_count": 0.0,
            "top_cluster_counts": {},
        }
    dominant_cluster_id, dominant_count = counts.most_common(1)[0]
    count_values = [int(value) for value in counts.values()]
    return {
        "dominant_cluster_id": dominant_cluster_id,
        "dominant_cluster_count": int(dominant_count),
        "dominant_cluster_fraction": float(dominant_count / len(clusters)),
        "cluster_gini": _gini(count_values),
        "mean_per_cluster_count": _mean(count_values),
        "top_cluster_counts": {key: int(value) for key, value in counts.most_common(5)},
    }


def _top_candidate_movers(
    current_rows: Sequence[dict[str, object]],
    shadow_rows: Sequence[dict[str, object]],
    *,
    n_examples: int,
) -> list[dict[str, object]]:
    current_rank_by_id = {_row_id(row): int(_to_float(row.get("rank", idx + 1), default=idx + 1)) for idx, row in enumerate(current_rows)}
    movers = []
    for row in shadow_rows:
        row_id = _row_id(row)
        current_rank = current_rank_by_id.get(row_id)
        if current_rank is None:
            continue
        shadow_rank = int(_to_float(row.get("rank", 0), default=0.0))
        movers.append(
            {
                "worker_id": row.get("worker_id", row_id),
                "sample_id": row.get("sample_id", row_id),
                "label": row.get("label", ""),
                "current_rank": current_rank,
                "shadow_rank": shadow_rank,
                "rank_improvement": current_rank - shadow_rank,
                "shadow_score": row.get("shadow_score", ""),
                "quality_score": row.get("quality_score", ""),
                "new_cluster_id": row.get("new_cluster_id", ""),
                "reason_code": row.get("reason_code", ""),
                "token_nll_p95": row.get("token_nll_p95", ""),
            }
        )
    return sorted(movers, key=lambda row: (-int(row["rank_improvement"]), int(row["shadow_rank"])))[:n_examples]


def _row_id(row: dict[str, object]) -> str:
    return str(row.get("sample_id") or row.get("worker_id") or row.get("candidate_index") or "")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _column(rows: Sequence[dict[str, object]], name: str, *, default: float = 0.0) -> np.ndarray:
    return np.asarray([_to_float(row.get(name, default), default=default) for row in rows], dtype=float)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array, dtype=float)
    lo = float(np.min(array[finite]))
    hi = float(np.max(array[finite]))
    span = hi - lo
    if span < 1.0e-12:
        return np.zeros_like(array, dtype=float)
    return np.nan_to_num((array - lo) / span, nan=0.0, posinf=1.0, neginf=0.0)


def _mean(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _to_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _to_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _gini(values: Sequence[float | int]) -> float:
    numbers = sorted(float(value) for value in values if float(value) >= 0.0)
    if not numbers:
        return 0.0
    total = sum(numbers)
    if total <= 0.0:
        return 0.0
    n = len(numbers)
    weighted = sum((idx + 1) * value for idx, value in enumerate(numbers))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Shadow ranking config must include a '{key}' object.")
    return value
