from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from marginal_value.eval.ablation_eval import average_precision_at_k, cluster_diversity_at_k, ndcg_at_k, precision_at_k
from marginal_value.logging_utils import log_event


DEFAULT_K_VALUES = [10, 50, 100, 200]
DEFAULT_LOW_QUALITY_THRESHOLD = 0.45
DEFAULT_CLUSTER_BONUS_WEIGHT = 0.25

GLOBAL_FORBIDDEN_SCORE_FEATURES = {
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


def run_score_calibration_eval(
    config: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    allow_local_execution: bool = False,
) -> dict[str, Any]:
    validate_score_calibration_config(config, allow_local_execution=allow_local_execution)
    artifacts = config["artifacts"]
    output = Path(output_dir) if output_dir is not None else Path(artifacts["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    eval_config = config.get("eval", {})
    k_values = [int(value) for value in eval_config.get("k_values", DEFAULT_K_VALUES)]
    low_quality_threshold = float(eval_config.get("low_quality_threshold", DEFAULT_LOW_QUALITY_THRESHOLD))
    primary_k = int(eval_config.get("primary_k", 100 if 100 in k_values else max(k_values)))

    log_event("score_calibration", "start", output_dir=str(output), dataset_count=len(artifacts["datasets"]))
    dataset_reports: dict[str, Any] = {}
    for dataset in artifacts["datasets"]:
        dataset_name = str(dataset["name"])
        dataset_path = Path(str(dataset["path"]))
        log_event("score_calibration", "dataset_read_start", dataset=dataset_name, path=str(dataset_path))
        rows = _read_csv(dataset_path)
        log_event("score_calibration", "dataset_read_done", dataset=dataset_name, n_rows=len(rows))
        dataset_reports[dataset_name] = evaluate_score_calibration_dataset(
            rows,
            dataset_config=dataset,
            k_values=k_values,
            low_quality_threshold=low_quality_threshold,
        )

    aggregate = _aggregate_dataset_reports(dataset_reports, primary_k=primary_k)
    report = {
        "mode": "full",
        "k_values": k_values,
        "primary_k": primary_k,
        "low_quality_threshold": low_quality_threshold,
        "datasets": dataset_reports,
        "aggregate": aggregate,
    }
    report_path = output / "score_calibration_report_full.json"
    summary_path = output / "score_calibration_summary_full.csv"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_rows(summary_path, dataset_reports)

    result = {
        "dataset_count": len(dataset_reports),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "best_overall_variant": aggregate.get("best_overall_variant", ""),
        "best_common_variant": aggregate.get("best_common_variant", ""),
        f"best_overall_selection_score@{primary_k}": aggregate.get(f"best_overall_selection_score@{primary_k}", 0.0),
        f"best_common_selection_score@{primary_k}": aggregate.get(f"best_common_selection_score@{primary_k}", 0.0),
    }
    log_event("score_calibration", "done", **result)
    return result


def evaluate_score_calibration_dataset(
    rows: Sequence[dict[str, object]],
    *,
    dataset_config: dict[str, Any],
    k_values: Iterable[int],
    low_quality_threshold: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty.")
    ks = [int(value) for value in k_values]
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("k_values must contain positive integers.")
    if not 0.0 <= low_quality_threshold <= 1.0:
        raise ValueError("low_quality_threshold must be in [0, 1].")

    dataset_name = str(dataset_config.get("name", "dataset"))
    forbidden = _dataset_forbidden_features(dataset_config)
    cluster_column = str(dataset_config.get("cluster_column", "new_cluster_id"))
    labels = np.asarray([_to_int(row.get("label", 0)) for row in rows], dtype=int)
    clusters = _cluster_values(rows, cluster_column=cluster_column)
    artifacts = np.asarray([_to_int(row.get("is_artifact", 0)) for row in rows], dtype=int)
    redundancies = np.asarray([_to_int(row.get("is_redundant", 0)) for row in rows], dtype=int)
    qualities = _column(rows, "quality_score", default=1.0)

    score_inputs = _build_score_recipes(rows, forbidden_features=forbidden)
    variants: dict[str, dict[str, Any]] = {}
    leaking_features: dict[str, list[str]] = {}
    for name, recipe in score_inputs.items():
        used_features = list(recipe["used_features"])
        forbidden_used = sorted(set(used_features) & forbidden)
        if forbidden_used:
            leaking_features[name] = forbidden_used
            continue
        scores = np.asarray(recipe["scores"], dtype=float)
        if scores.shape != labels.shape or not np.all(np.isfinite(scores)):
            continue
        order = np.argsort(-scores, kind="mergesort")
        variants[name] = _summarize_scores(
            scores=scores,
            order=order,
            labels=labels,
            clusters=clusters,
            qualities=qualities,
            artifacts=artifacts,
            redundancies=redundancies,
            k_values=ks,
            low_quality_threshold=low_quality_threshold,
            used_features=used_features,
            order_source="score_sort",
        )

        if _allow_cluster_aware_variant(dataset_config, cluster_column, forbidden):
            reranked_order = _cluster_aware_order(
                scores,
                clusters,
                cluster_bonus_weight=float(dataset_config.get("cluster_bonus_weight", DEFAULT_CLUSTER_BONUS_WEIGHT)),
            )
            variants[f"{name}_cluster_aware"] = _summarize_scores(
                scores=scores,
                order=reranked_order,
                labels=labels,
                clusters=clusters,
                qualities=qualities,
                artifacts=artifacts,
                redundancies=redundancies,
                k_values=ks,
                low_quality_threshold=low_quality_threshold,
                used_features=used_features + [cluster_column],
                order_source="cluster_aware",
            )

    primary_k = 100 if 100 in ks else max(ks)
    return {
        "dataset_name": dataset_name,
        "n_rows": len(rows),
        "label_counts": {str(key): int(value) for key, value in sorted(Counter(labels.tolist()).items())},
        "cluster_column": cluster_column,
        "forbidden_score_features": sorted(forbidden),
        "leakage_audit": {
            "leaking_variant_features": leaking_features,
            "score_feature_intersections_empty": not leaking_features,
        },
        "variants": variants,
        f"best_by_selection_score@{primary_k}": _best_variant_by_selection_score(variants, primary_k),
    }


def validate_score_calibration_config(config: dict[str, Any], *, allow_local_execution: bool = False) -> None:
    execution = _required_mapping(config, "execution")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal" and not allow_local_execution:
        raise ValueError("Score calibration evaluation must be dispatched through Modal.")
    output_dir = Path(str(artifacts.get("output_dir", "")))
    if not output_dir.is_absolute():
        raise ValueError("artifacts.output_dir must be absolute.")
    datasets = artifacts.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("artifacts.datasets must be a non-empty list.")
    seen_names: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Each score calibration dataset must be an object.")
        name = str(dataset.get("name", "")).strip()
        if not name:
            raise ValueError("Each score calibration dataset must include a name.")
        if name in seen_names:
            raise ValueError(f"Duplicate score calibration dataset name: {name}")
        seen_names.add(name)
        path = Path(str(dataset.get("path", "")))
        if not path.is_absolute():
            raise ValueError(f"Dataset '{name}' path must be absolute.")

    eval_config = config.get("eval", {})
    k_values = eval_config.get("k_values", DEFAULT_K_VALUES)
    if not k_values or any(int(value) <= 0 for value in k_values):
        raise ValueError("eval.k_values must contain positive integers.")
    low_quality_threshold = float(eval_config.get("low_quality_threshold", DEFAULT_LOW_QUALITY_THRESHOLD))
    if not 0.0 <= low_quality_threshold <= 1.0:
        raise ValueError("eval.low_quality_threshold must be in [0, 1].")


def load_score_calibration_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_score_recipes(
    rows: Sequence[dict[str, object]],
    *,
    forbidden_features: set[str],
) -> dict[str, dict[str, Any]]:
    recipes: dict[str, dict[str, Any]] = {}

    if _has_columns(rows, ["final_score"]):
        recipes["current_final_score"] = {
            "scores": _column(rows, "final_score"),
            "used_features": ["final_score"],
        }

    grammar_recipe = _grammar_surprisal_recipe(rows)
    if grammar_recipe is not None:
        recipes["grammar_surprisal_mix"] = grammar_recipe
        quality = np.clip(_column(rows, "quality_score", default=1.0), 0.0, 1.0)
        recipes["quality_gated_grammar"] = {
            "scores": grammar_recipe["scores"] * quality,
            "used_features": grammar_recipe["used_features"] + ["quality_score"],
        }
        token_support = _token_support_penalty(rows)
        if token_support is not None:
            recipes["grammar_quality_token_support_penalized"] = {
                "scores": grammar_recipe["scores"] * quality * token_support["scores"],
                "used_features": grammar_recipe["used_features"] + ["quality_score"] + token_support["used_features"],
            }

    novelty = _first_available_column(rows, ["old_novelty_score", "old_knn_distance"])
    support = _first_available_column(rows, ["new_density_score", "new_batch_density"])
    if novelty is not None and support is not None:
        novelty_scores = _normalize(novelty["values"])
        support_scores = _normalize(support["values"])
        quality = np.clip(_column(rows, "quality_score", default=1.0), 0.0, 1.0)
        base_scores = novelty_scores * support_scores * quality
        used_features = [novelty["name"], support["name"], "quality_score"]
        recipes["novelty_quality_support"] = {
            "scores": base_scores,
            "used_features": used_features,
        }
        redundancy = _redundancy_penalty(rows)
        if redundancy is not None:
            recipes["novelty_quality_support_minus_redundancy"] = {
                "scores": base_scores * redundancy["scores"],
                "used_features": used_features + redundancy["used_features"],
            }

    return {
        name: recipe
        for name, recipe in recipes.items()
        if not (set(recipe["used_features"]) & forbidden_features)
    }


def _grammar_surprisal_recipe(rows: Sequence[dict[str, object]]) -> dict[str, Any] | None:
    required = ["token_nll_p95", "transition_nll_p95"]
    if not _has_columns(rows, required):
        return None
    token_nll = _normalize(_column(rows, "token_nll_p95"))
    transition_nll = _normalize(_column(rows, "transition_nll_p95"))
    used_features = ["token_nll_p95", "transition_nll_p95"]
    if _has_columns(rows, ["longest_unseen_phrase_len"]):
        longest_unseen = _normalize(_column(rows, "longest_unseen_phrase_len"))
        used_features.append("longest_unseen_phrase_len")
    else:
        longest_unseen = np.zeros(len(rows), dtype=float)
    scores = _normalize(0.65 * token_nll + 0.25 * transition_nll + 0.10 * longest_unseen)
    return {"scores": scores, "used_features": used_features}


def _token_support_penalty(rows: Sequence[dict[str, object]]) -> dict[str, Any] | None:
    if not _has_columns(rows, ["token_duplicate_fraction", "token_neighborhood_density"]):
        return None
    duplicate_fraction = _normalize(_column(rows, "token_duplicate_fraction"))
    density = _normalize(_column(rows, "token_neighborhood_density"))
    scores = np.clip(1.0 - 0.50 * duplicate_fraction - 0.25 * density, 0.0, 1.0)
    return {"scores": scores, "used_features": ["token_duplicate_fraction", "token_neighborhood_density"]}


def _redundancy_penalty(rows: Sequence[dict[str, object]]) -> dict[str, Any] | None:
    if _has_columns(rows, ["redundancy_penalty"]):
        penalty = _normalize(_column(rows, "redundancy_penalty"))
        return {"scores": np.clip(1.0 - 0.50 * penalty, 0.0, 1.0), "used_features": ["redundancy_penalty"]}
    token_support = _token_support_penalty(rows)
    return token_support


def _summarize_scores(
    *,
    scores: np.ndarray,
    order: Sequence[int],
    labels: np.ndarray,
    clusters: np.ndarray,
    qualities: np.ndarray,
    artifacts: np.ndarray,
    redundancies: np.ndarray,
    k_values: Sequence[int],
    low_quality_threshold: float,
    used_features: Sequence[str],
    order_source: str,
) -> dict[str, Any]:
    ordered_labels = labels[list(order)]
    ordered_clusters = clusters[list(order)]
    ordered_qualities = qualities[list(order)]
    ordered_artifacts = artifacts[list(order)]
    ordered_redundancies = redundancies[list(order)]
    ordered_scores = scores[list(order)]
    total_clusters = max(1, len(set(clusters.tolist())))
    metrics: dict[str, float] = {
        "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        "positive_rate": float(np.mean(labels > 0)) if len(labels) else 0.0,
    }
    for k in k_values:
        k_eff = min(int(k), len(order))
        top_labels = ordered_labels[:k_eff]
        top_clusters = ordered_clusters[:k_eff]
        top_qualities = ordered_qualities[:k_eff]
        top_artifacts = ordered_artifacts[:k_eff]
        top_redundancies = ordered_redundancies[:k_eff]
        unique_cluster_count = len(set(top_clusters.tolist()))
        low_quality_rate = float(np.mean(top_qualities < low_quality_threshold)) if k_eff else 0.0
        artifact_rate = float(np.mean(top_artifacts > 0)) if k_eff else 0.0
        redundancy_rate = float(np.mean(top_redundancies > 0)) if k_eff else 0.0
        ndcg = ndcg_at_k(ordered_labels, k_eff)
        cluster_diversity = cluster_diversity_at_k(ordered_labels, ordered_clusters, k_eff)
        metrics[f"precision@{k}"] = precision_at_k(ordered_labels, k_eff)
        metrics[f"ap@{k}"] = average_precision_at_k(ordered_labels, k_eff)
        metrics[f"ndcg@{k}"] = ndcg
        metrics[f"cluster_diversity@{k}"] = cluster_diversity
        metrics[f"unique_cluster_count@{k}"] = float(unique_cluster_count)
        metrics[f"cluster_repeat_count@{k}"] = float(k_eff - unique_cluster_count)
        metrics[f"cluster_coverage_fraction@{k}"] = float(unique_cluster_count / total_clusters)
        metrics[f"mean_quality@{k}"] = float(np.mean(top_qualities)) if k_eff else 0.0
        metrics[f"low_quality_rate@{k}"] = low_quality_rate
        metrics[f"artifact_rate@{k}"] = artifact_rate
        metrics[f"redundancy_rate@{k}"] = redundancy_rate
        metrics[f"mean_score@{k}"] = float(np.mean(ordered_scores[:k_eff])) if k_eff else 0.0
        metrics[f"selection_score@{k}"] = (
            ndcg
            + 0.50 * cluster_diversity
            - artifact_rate
            - 0.50 * redundancy_rate
            - low_quality_rate
        )
    return {
        "eligible": True,
        "order_source": order_source,
        "used_features": list(used_features),
        "metrics": metrics,
    }


def _cluster_aware_order(
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    cluster_bonus_weight: float,
) -> list[int]:
    base_scores = np.asarray(scores, dtype=float)
    remaining = np.ones(len(base_scores), dtype=bool)
    selected: list[int] = []
    seen_clusters: set[str] = set()
    while len(selected) < len(base_scores):
        adjusted = base_scores.copy()
        for idx, cluster in enumerate(clusters.tolist()):
            if str(cluster) not in seen_clusters:
                adjusted[idx] += cluster_bonus_weight
        adjusted[~remaining] = -np.inf
        best_idx = int(np.argmax(adjusted))
        if not np.isfinite(adjusted[best_idx]):
            break
        selected.append(best_idx)
        remaining[best_idx] = False
        seen_clusters.add(str(clusters[best_idx]))
    return selected


def _aggregate_dataset_reports(dataset_reports: dict[str, Any], *, primary_k: int) -> dict[str, Any]:
    variant_scores: dict[str, list[float]] = defaultdict(list)
    for report in dataset_reports.values():
        for variant_name, variant in report.get("variants", {}).items():
            metrics = variant.get("metrics", {})
            key = f"selection_score@{primary_k}"
            if key in metrics:
                variant_scores[variant_name].append(float(metrics[key]))
    aggregate_rows = {
        name: {
            "dataset_count": len(values),
            f"mean_selection_score@{primary_k}": float(np.mean(values)) if values else 0.0,
        }
        for name, values in sorted(variant_scores.items())
    }
    if not aggregate_rows:
        return {
            "variants": {},
            "best_available_variant": "",
            "best_common_variant": "",
            "best_overall_variant": "",
            "best_common_dataset_count": 0,
            f"best_available_selection_score@{primary_k}": 0.0,
            f"best_common_selection_score@{primary_k}": 0.0,
            f"best_overall_selection_score@{primary_k}": 0.0,
        }
    best_available_name = max(
        aggregate_rows,
        key=lambda name: (aggregate_rows[name][f"mean_selection_score@{primary_k}"], aggregate_rows[name]["dataset_count"]),
    )
    max_dataset_count = max(int(row["dataset_count"]) for row in aggregate_rows.values())
    common_candidates = {
        name: row for name, row in aggregate_rows.items() if int(row["dataset_count"]) == max_dataset_count
    }
    best_common_name = max(
        common_candidates,
        key=lambda name: common_candidates[name][f"mean_selection_score@{primary_k}"],
    )
    return {
        "variants": aggregate_rows,
        "best_available_variant": best_available_name,
        f"best_available_selection_score@{primary_k}": aggregate_rows[best_available_name][
            f"mean_selection_score@{primary_k}"
        ],
        "best_common_variant": best_common_name,
        "best_common_dataset_count": max_dataset_count,
        f"best_common_selection_score@{primary_k}": aggregate_rows[best_common_name][
            f"mean_selection_score@{primary_k}"
        ],
        "best_overall_variant": best_common_name,
        f"best_overall_selection_score@{primary_k}": aggregate_rows[best_common_name][
            f"mean_selection_score@{primary_k}"
        ],
    }


def _write_summary_rows(path: Path, dataset_reports: dict[str, Any]) -> None:
    rows: list[dict[str, object]] = []
    for dataset_name, report in sorted(dataset_reports.items()):
        for variant_name, variant in sorted(report.get("variants", {}).items()):
            row: dict[str, object] = {
                "dataset": dataset_name,
                "variant": variant_name,
                "order_source": variant.get("order_source", ""),
                "used_features": "|".join(str(value) for value in variant.get("used_features", [])),
            }
            row.update(variant.get("metrics", {}))
            rows.append(row)
    _write_rows(path, rows)


def _best_variant_by_selection_score(variants: dict[str, dict[str, Any]], primary_k: int) -> str:
    if not variants:
        return ""
    key = f"selection_score@{primary_k}"
    return max(variants, key=lambda name: float(variants[name].get("metrics", {}).get(key, -float("inf"))))


def _allow_cluster_aware_variant(dataset_config: dict[str, Any], cluster_column: str, forbidden: set[str]) -> bool:
    return bool(dataset_config.get("allow_cluster_aware_variants", False)) and cluster_column not in forbidden


def _dataset_forbidden_features(dataset_config: dict[str, Any]) -> set[str]:
    return GLOBAL_FORBIDDEN_SCORE_FEATURES | {str(value) for value in dataset_config.get("forbidden_score_features", [])}


def _first_available_column(rows: Sequence[dict[str, object]], names: Sequence[str]) -> dict[str, Any] | None:
    for name in names:
        if _has_columns(rows, [name]):
            return {"name": name, "values": _column(rows, name)}
    return None


def _cluster_values(rows: Sequence[dict[str, object]], *, cluster_column: str) -> np.ndarray:
    if _has_columns(rows, [cluster_column]):
        return np.asarray([str(row.get(cluster_column, idx)) for idx, row in enumerate(rows)], dtype=object)
    return np.asarray([str(idx) for idx in range(len(rows))], dtype=object)


def _has_columns(rows: Sequence[dict[str, object]], columns: Sequence[str]) -> bool:
    return all(all(str(row.get(column, "")).strip() != "" for row in rows) for column in columns)


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


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Score calibration config must include a '{key}' object.")
    return value
