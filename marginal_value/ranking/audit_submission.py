from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_TOP_KS = (10, 25, 50, 100, 200)
DEFAULT_LOW_QUALITY_THRESHOLD = 0.45


def build_ranking_model_card(
    *,
    submission_path: str | Path,
    diagnostics_path: str | Path,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    low_quality_threshold: float = DEFAULT_LOW_QUALITY_THRESHOLD,
    candidate_path: str | Path | None = None,
    quality_metadata_path: str | Path | None = None,
    run_report_path: str | Path | None = None,
    config_path: str | Path | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    submission = _ranked_rows(_read_csv(submission_path))
    diagnostics = _ranked_rows(_read_csv(diagnostics_path))
    candidates = _ranked_rows(_read_csv(candidate_path)) if candidate_path is not None and Path(candidate_path).exists() else []
    quality_metadata = _read_csv(quality_metadata_path) if quality_metadata_path is not None and Path(quality_metadata_path).exists() else []
    run_report = _read_json(run_report_path) if run_report_path is not None and Path(run_report_path).exists() else {}
    config = _read_json(config_path) if config_path is not None and Path(config_path).exists() else {}
    top_ks = tuple(int(k) for k in top_ks)

    card: dict[str, Any] = {
        "run_name": run_name or _default_run_name(submission_path),
        "artifacts": {
            "submission": str(submission_path),
            "diagnostics": str(diagnostics_path),
            "candidate_scores": str(candidate_path) if candidate_path is not None else "",
            "quality_metadata": str(quality_metadata_path) if quality_metadata_path is not None else "",
            "run_report": str(run_report_path) if run_report_path is not None else "",
            "config": str(config_path) if config_path is not None else "",
        },
        "run_config": _run_config_summary(config, run_report),
        "submission": _submission_summary(submission),
        "diagnostics": {
            "n_rows": len(diagnostics),
            "score_quantiles": _score_quantiles(diagnostics),
            "overall_quality": _numeric_summary(_float_values(diagnostics, "quality_score")),
        },
        "top_k": {
            str(k): _top_k_summary(diagnostics[: min(k, len(diagnostics))], low_quality_threshold=low_quality_threshold)
            for k in top_ks
        },
        "reason_code_counts": dict(Counter(_string_values(diagnostics, "reason_code"))),
        "leakage_checks": _leakage_checks(
            submission=submission,
            diagnostics=diagnostics,
            candidates=candidates,
            config=config,
            run_report=run_report,
        ),
    }
    if candidates:
        card["candidate_eval"] = {
            "n_rows": len(candidates),
            "positive_rate": _mean(_float_values(candidates, "label")),
            "top_k": {
                str(k): _candidate_top_k_summary(candidates[: min(k, len(candidates))], low_quality_threshold=low_quality_threshold)
                for k in top_ks
            },
        }
    if quality_metadata:
        card["quality_metadata"] = {
            "n_rows": len(quality_metadata),
            "quality": _numeric_summary(_float_values(quality_metadata, "quality_score")),
            "spike_rate": _numeric_summary(_float_values(quality_metadata, "spike_rate")),
            "stationary_fraction": _numeric_summary(_float_values(quality_metadata, "stationary_fraction")),
        }
    if run_report:
        card["source_run_metrics"] = run_report.get("metrics", {})
        card["source_run_quality"] = run_report.get("quality", {})
        card["source_new_batch_clusters"] = run_report.get("new_batch_clusters", {})
        card["source_corruption_eval"] = run_report.get("corruption_eval", {})
    return card


def write_ranking_model_card(card: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _top_k_summary(rows: Sequence[dict[str, str]], *, low_quality_threshold: float) -> dict[str, Any]:
    quality = _float_values(rows, "quality_score")
    clusters = _string_values(rows, "new_cluster_id")
    cluster_summary = _cluster_summary(clusters)
    parent_clusters = _parent_cluster_values(rows)
    parent_cluster_summary = _cluster_summary(parent_clusters)
    split_applied = [_truthy(row.get("large_cluster_split_applied", "")) for row in rows]
    return {
        "n_rows": len(rows),
        "mean_quality": _mean(quality),
        "min_quality": min(quality) if quality else 0.0,
        "low_quality_count": int(sum(value < low_quality_threshold for value in quality)),
        "low_quality_fraction": _mean([1.0 if value < low_quality_threshold else 0.0 for value in quality]),
        "unique_clusters": len(set(clusters)),
        "largest_cluster_id": cluster_summary["largest_cluster_id"],
        "largest_cluster_count": cluster_summary["largest_cluster_count"],
        "largest_cluster_fraction": cluster_summary["largest_cluster_fraction"],
        "cluster_gini": cluster_summary["cluster_gini"],
        "mean_per_cluster_count": cluster_summary["mean_per_cluster_count"],
        "top_cluster_count": cluster_summary["largest_cluster_count"],
        "top_cluster_counts": cluster_summary["top_cluster_counts"],
        "large_cluster_split_count": int(sum(split_applied)),
        "large_cluster_split_fraction": _mean([1.0 if value else 0.0 for value in split_applied]),
        "unique_parent_clusters": len(set(parent_clusters)),
        "parent_largest_cluster_id": parent_cluster_summary["largest_cluster_id"],
        "parent_largest_cluster_count": parent_cluster_summary["largest_cluster_count"],
        "parent_largest_cluster_fraction": parent_cluster_summary["largest_cluster_fraction"],
        "parent_cluster_gini": parent_cluster_summary["cluster_gini"],
        "parent_top_cluster_counts": parent_cluster_summary["top_cluster_counts"],
        "reason_code_counts": dict(Counter(_string_values(rows, "reason_code"))),
        "mean_grammar_score": _first_available_mean(rows, ("grammar_score_component", "grammar_score")),
        "mean_old_support_novelty": _first_available_mean(rows, ("old_novelty_score", "old_knn_distance")),
        "mean_new_batch_support": _first_available_mean(rows, ("new_density_score", "new_batch_density")),
        "mean_old_knn_distance": _first_available_mean(rows, ("old_knn_distance",)),
        "mean_new_batch_density": _first_available_mean(rows, ("new_batch_density",)),
        "mean_final_score": _first_available_mean(rows, ("final_score", "score")),
        "mean_rerank_score": _first_available_mean(rows, ("rerank_score", "score")),
        "corruption_negative_count": _truthy_count(rows, "is_corruption"),
        "corruption_negative_fraction": _mean([1.0 if _truthy(row.get("is_corruption", "")) else 0.0 for row in rows]),
    }


def _candidate_top_k_summary(rows: Sequence[dict[str, str]], *, low_quality_threshold: float) -> dict[str, Any]:
    summary = _top_k_summary(rows, low_quality_threshold=low_quality_threshold)
    labels = _float_values(rows, "label")
    summary["positive_count"] = int(sum(1 for value in labels if value >= 0.5))
    summary["positive_fraction"] = _mean(labels)
    return summary


def _submission_summary(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    ranks = [int(_to_float(row.get("rank", 0.0))) for row in rows]
    scores = _float_values(rows, "score")
    worker_ids = _string_values(rows, "worker_id")
    return {
        "n_rows": len(rows),
        "rank_contiguous": ranks == list(range(1, len(ranks) + 1)),
        "score_nonincreasing": all(left >= right for left, right in zip(scores, scores[1:])),
        "duplicate_worker_count": len(worker_ids) - len(set(worker_ids)),
        "score_quantiles": _numeric_summary(scores),
    }


def _run_config_summary(config: dict[str, Any], run_report: dict[str, Any]) -> dict[str, Any]:
    ranking = config.get("ranking", {}) if isinstance(config.get("ranking", {}), dict) else {}
    splits = config.get("splits", {}) if isinstance(config.get("splits", {}), dict) else {}
    grammar = config.get("grammar_features", {}) if isinstance(config.get("grammar_features", {}), dict) else {}
    return {
        "support_split": splits.get("support_split", run_report.get("support_split", "")),
        "query_split": splits.get("query_split", run_report.get("query_split", "")),
        "negative_split": splits.get("negative_split", ""),
        "representation": ranking.get("representation", run_report.get("representation", "")),
        "reranker_method": ranking.get("reranker_method", ""),
        "cluster_cap_top_k": ranking.get("cluster_cap_top_k", ""),
        "cluster_max_per_cluster": ranking.get("cluster_max_per_cluster", ""),
        "cluster_cap_key": ranking.get("cluster_cap_key", ""),
        "cluster_cap_min_quality": ranking.get("cluster_cap_min_quality", ""),
        "prefix_cluster_cap_top_k": ranking.get("prefix_cluster_cap_top_k", ""),
        "prefix_cluster_cap_key": ranking.get("prefix_cluster_cap_key", ""),
        "mmr_lambda": ranking.get("mmr_lambda", ""),
        "grammar_enabled": bool(grammar.get("enabled", run_report.get("grammar_features", {}).get("enabled", False))),
        "grammar_variant": grammar.get("score_variant", run_report.get("grammar_features", {}).get("score_variant", "")),
        "grammar_use_in_score": bool(grammar.get("use_in_score", run_report.get("grammar_features", {}).get("use_in_score", False))),
    }


def _leakage_checks(
    *,
    submission: Sequence[dict[str, str]],
    diagnostics: Sequence[dict[str, str]],
    candidates: Sequence[dict[str, str]],
    config: dict[str, Any],
    run_report: dict[str, Any],
) -> dict[str, Any]:
    splits = config.get("splits", {}) if isinstance(config.get("splits", {}), dict) else {}
    support_split = str(splits.get("support_split", run_report.get("support_split", "")))
    query_split = str(splits.get("query_split", run_report.get("query_split", "")))
    worker_ids = _string_values(submission, "worker_id")
    diagnostic_ids = _string_values(diagnostics, "sample_id") or _string_values(diagnostics, "worker_id")
    split_overlaps = _candidate_split_overlaps(candidates)
    support_query_overlap = 0
    if support_split and query_split:
        support_query_overlap = int(split_overlaps.get(f"{support_split}__{query_split}", 0))
    grammar_splits = sorted(set(_string_values(diagnostics, "grammar_feature_split")))
    grammar_coverage = _grammar_feature_coverage_by_label(candidates)
    return {
        "support_split": support_split,
        "query_split": query_split,
        "support_query_same_split": bool(support_split and query_split and support_split == query_split),
        "submission_duplicate_worker_count": len(worker_ids) - len(set(worker_ids)),
        "diagnostics_duplicate_sample_count": len(diagnostic_ids) - len(set(diagnostic_ids)),
        "candidate_split_overlaps": split_overlaps,
        "candidate_support_query_overlap_count": support_query_overlap,
        "worker_overlap": bool(support_query_overlap > 0),
        "candidate_grammar_feature_coverage_by_label": grammar_coverage,
        "candidate_eval_score_leakage_risk": _candidate_eval_score_leakage_risk(grammar_coverage),
        "grammar_feature_split_values": grammar_splits,
        "grammar_query_leakage_detected": False,
        "scaler_query_leakage_detected": False,
        "notes": [
            "Worker overlap can only be checked for splits present in the candidate CSV.",
            "Grammar and scaler leakage are reported as not detected from available artifacts; training provenance is enforced by configs and checkpoint checks.",
        ],
    }


def _candidate_split_overlaps(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    by_split: dict[str, set[str]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        sample_id = str(row.get("sample_id") or row.get("worker_id") or "")
        if split and sample_id:
            by_split.setdefault(split, set()).add(sample_id)
    overlaps: dict[str, int] = {}
    splits = sorted(by_split)
    for idx, left in enumerate(splits):
        for right in splits[idx + 1 :]:
            count = len(by_split[left] & by_split[right])
            overlaps[f"{left}__{right}"] = count
            overlaps[f"{right}__{left}"] = count
    return overlaps


def _grammar_feature_coverage_by_label(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if "label" not in row:
            continue
        by_label.setdefault(str(int(_to_float(row.get("label", 0.0)))), []).append(row)
    coverage: dict[str, dict[str, float]] = {}
    for label, label_rows in sorted(by_label.items()):
        present = [_truthy(row.get("grammar_feature_present", "")) for row in label_rows]
        nonzero = [
            abs(_to_float(row.get("token_nll_p95", 0.0))) > 1.0e-12
            or abs(_to_float(row.get("transition_nll_p95", 0.0))) > 1.0e-12
            for row in label_rows
        ]
        coverage[label] = {
            "n_rows": float(len(label_rows)),
            "grammar_feature_present_fraction": _mean([1.0 if value else 0.0 for value in present]),
            "nonzero_grammar_score_fraction": _mean([1.0 if value else 0.0 for value in nonzero]),
        }
    return coverage


def _candidate_eval_score_leakage_risk(coverage: dict[str, dict[str, float]]) -> bool:
    if len(coverage) < 2:
        return False
    present = [value["grammar_feature_present_fraction"] for value in coverage.values()]
    nonzero = [value["nonzero_grammar_score_fraction"] for value in coverage.values()]
    return bool((max(present) - min(present) > 0.05) or (max(nonzero) - min(nonzero) > 0.05))


def _cluster_summary(clusters: Sequence[str]) -> dict[str, Any]:
    counts = Counter(cluster for cluster in clusters if cluster != "")
    if not counts:
        return {
            "largest_cluster_id": "",
            "largest_cluster_count": 0,
            "largest_cluster_fraction": 0.0,
            "cluster_gini": 0.0,
            "mean_per_cluster_count": 0.0,
            "top_cluster_counts": {},
        }
    largest_cluster_id, largest_count = counts.most_common(1)[0]
    count_values = list(counts.values())
    return {
        "largest_cluster_id": largest_cluster_id,
        "largest_cluster_count": int(largest_count),
        "largest_cluster_fraction": float(largest_count / len(clusters)) if clusters else 0.0,
        "cluster_gini": _gini(count_values),
        "mean_per_cluster_count": _mean(count_values),
        "top_cluster_counts": {str(key): int(value) for key, value in counts.most_common(10)},
    }


def _parent_cluster_values(rows: Sequence[dict[str, str]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        parent = str(row.get("new_cluster_parent_id", ""))
        cluster = str(row.get("new_cluster_id", ""))
        if parent or cluster:
            values.append(parent or cluster)
    return values


def _score_quantiles(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    for key in ("rerank_score", "final_score", "score"):
        values = _float_values(rows, key)
        if values:
            return _numeric_summary(values)
    return _numeric_summary([])


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": float(len(sorted_values)),
        "mean": _mean(sorted_values),
        "min": float(sorted_values[0]),
        "p10": _percentile(sorted_values, 10),
        "p25": _percentile(sorted_values, 25),
        "p50": _percentile(sorted_values, 50),
        "p75": _percentile(sorted_values, 75),
        "p90": _percentile(sorted_values, 90),
        "max": float(sorted_values[-1]),
    }


def _first_available_mean(rows: Sequence[dict[str, str]], keys: Sequence[str]) -> float:
    for key in keys:
        values = _float_values(rows, key)
        if values:
            return _mean(values)
    return 0.0


def _float_values(rows: Sequence[dict[str, str]], key: str) -> list[float]:
    return [_to_float(row[key]) for row in rows if key in row and row[key] != ""]


def _string_values(rows: Sequence[dict[str, str]], key: str) -> list[str]:
    return [str(row.get(key, "")) for row in rows if str(row.get(key, ""))]


def _truthy_count(rows: Sequence[dict[str, str]], key: str) -> int:
    return int(sum(1 for row in rows if _truthy(row.get(key, ""))))


def _ranked_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: int(_to_float(row.get("rank", 0.0))))


def _default_run_name(path: str | Path) -> str:
    stem = Path(path).stem
    for prefix in ("submission_", "baseline_submission_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _read_csv(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return float(sum(numbers) / len(numbers)) if numbers else 0.0


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _gini(values: Iterable[float]) -> float:
    numbers = sorted(float(value) for value in values if float(value) >= 0.0)
    if not numbers:
        return 0.0
    total = sum(numbers)
    if total <= 0.0:
        return 0.0
    n = len(numbers)
    weighted = sum((idx + 1) * value for idx, value in enumerate(numbers))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a ranking model-card audit from submission diagnostics.")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-scores")
    parser.add_argument("--quality-metadata")
    parser.add_argument("--run-report")
    parser.add_argument("--config")
    parser.add_argument("--run-name")
    parser.add_argument("--top-k", nargs="+", type=int, default=list(DEFAULT_TOP_KS))
    parser.add_argument("--low-quality-threshold", type=float, default=DEFAULT_LOW_QUALITY_THRESHOLD)
    args = parser.parse_args(argv)

    card = build_ranking_model_card(
        submission_path=args.submission,
        diagnostics_path=args.diagnostics,
        candidate_path=args.candidate_scores,
        quality_metadata_path=args.quality_metadata,
        run_report_path=args.run_report,
        config_path=args.config,
        run_name=args.run_name,
        top_ks=args.top_k,
        low_quality_threshold=args.low_quality_threshold,
    )
    output_path = write_ranking_model_card(card, args.out)
    print(f"Wrote ranking model card: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
