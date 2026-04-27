from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from marginal_value.logging_utils import log_event


def audit_ranking_artifacts(
    *,
    submission_path: str | Path,
    diagnostics_path: str | Path,
    quality_metadata_path: str | Path,
    candidate_path: str | Path | None = None,
    top_ks: Iterable[int] = (10, 50, 100),
    low_quality_threshold: float = 0.45,
    n_examples: int = 25,
) -> dict[str, Any]:
    log_event(
        "ranking_audit",
        "read_start",
        submission_path=str(submission_path),
        diagnostics_path=str(diagnostics_path),
        quality_metadata_path=str(quality_metadata_path),
        candidate_path=str(candidate_path) if candidate_path is not None else "",
    )
    submission = _read_csv(submission_path)
    diagnostics = _read_csv(diagnostics_path)
    quality_metadata = _read_csv(quality_metadata_path)
    candidates = _read_csv(candidate_path) if candidate_path is not None and Path(candidate_path).exists() else []
    log_event(
        "ranking_audit",
        "read_done",
        n_submission=len(submission),
        n_diagnostics=len(diagnostics),
        n_quality=len(quality_metadata),
        n_candidates=len(candidates),
    )

    ranked_submission = sorted(submission, key=lambda row: int(float(row.get("rank", 0))))
    ranked_diagnostics = sorted(diagnostics, key=lambda row: int(float(row.get("rank", 0))))
    top_ks_tuple = tuple(int(k) for k in top_ks)
    report: dict[str, Any] = {
        "submission": _submission_summary(ranked_submission),
        "overall_quality": _quality_summary(_float_column(ranked_diagnostics, "quality_score")),
        "reason_code_counts": dict(Counter(_string_column(ranked_diagnostics, "reason_code"))),
        "top_k": {
            str(k): _rank_slice_summary(ranked_diagnostics[:k], low_quality_threshold=low_quality_threshold)
            for k in top_ks_tuple
        },
        "top_examples": ranked_diagnostics[:n_examples],
        "low_quality_examples": _low_quality_examples(
            ranked_diagnostics,
            low_quality_threshold=low_quality_threshold,
            n_examples=n_examples,
        ),
        "quality_metadata": {
            "n_rows": len(quality_metadata),
            "quality": _quality_summary(_float_column(quality_metadata, "quality_score")),
            "spike_rate": _quality_summary(_float_column(quality_metadata, "spike_rate")),
            "stationary_fraction": _quality_summary(_float_column(quality_metadata, "stationary_fraction")),
        },
    }
    if candidates:
        log_event("ranking_audit", "candidate_summary_start", n_candidates=len(candidates))
        report["candidate_eval"] = _candidate_eval_summary(
            candidates,
            top_ks=top_ks_tuple,
            low_quality_threshold=low_quality_threshold,
        )
    grammar_summary = _grammar_diagnostics_summary(
        ranked_diagnostics,
        low_quality_threshold=low_quality_threshold,
        n_examples=n_examples,
    )
    if grammar_summary is not None:
        report["grammar_diagnostics"] = grammar_summary
    log_event("ranking_audit", "audit_done", n_submission_rows=len(submission))
    return report


def write_audit_artifacts(report: dict[str, Any], output_dir: str | Path, *, suffix: str) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_event("ranking_audit", "write_start", output_dir=str(output), suffix=suffix)
    paths = {
        "report": output / f"baseline_audit_{suffix}.json",
        "top_examples": output / f"baseline_audit_top_examples_{suffix}.csv",
        "low_quality_examples": output / f"baseline_audit_low_quality_examples_{suffix}.csv",
        "top_grammar_examples": output / f"baseline_audit_top_grammar_examples_{suffix}.csv",
        "reason_counts": output / f"baseline_audit_reason_counts_{suffix}.csv",
    }
    paths["report"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_rows(paths["top_examples"], report.get("top_examples", []))
    _write_rows(paths["low_quality_examples"], report.get("low_quality_examples", []))
    _write_rows(paths["top_grammar_examples"], report.get("grammar_diagnostics", {}).get("top_surprisal_examples", []))
    _write_rows(
        paths["reason_counts"],
        [
            {"reason_code": reason, "count": count}
            for reason, count in sorted(report.get("reason_code_counts", {}).items())
        ],
    )
    log_event("ranking_audit", "write_done", report_path=str(paths["report"]))
    return paths


def run_ranking_audit(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    suffix = "smoke" if smoke else "full"
    artifacts = config["artifacts"]
    ranking_dir = Path(artifacts["ranking_dir"])
    output_dir = Path(artifacts.get("audit_dir", ranking_dir))
    log_event("ranking_audit", "start", smoke=smoke, ranking_dir=str(ranking_dir), output_dir=str(output_dir))
    report = audit_ranking_artifacts(
        submission_path=ranking_dir / f"baseline_submission_val_{suffix}.csv",
        diagnostics_path=ranking_dir / f"baseline_diagnostics_val_{suffix}.csv",
        candidate_path=ranking_dir / f"baseline_ranking_val_candidates_{suffix}.csv",
        quality_metadata_path=ranking_dir / f"baseline_quality_metadata_{suffix}.csv",
        top_ks=config.get("audit", {}).get("top_ks", [10, 50, 100]),
        low_quality_threshold=float(config.get("audit", {}).get("low_quality_threshold", 0.45)),
        n_examples=int(config.get("audit", {}).get("n_examples", 25)),
    )
    output_paths = write_audit_artifacts(report, output_dir, suffix=suffix)
    result = {
        "mode": suffix,
        "report_path": str(output_paths["report"]),
        "n_submission_rows": report["submission"]["n_rows"],
        "score_nonincreasing": report["submission"]["score_nonincreasing"],
        "top10_mean_quality": report["top_k"].get("10", {}).get("mean_quality", 0.0),
        "top100_low_quality_count": report["top_k"].get("100", {}).get("low_quality_count", 0),
        "reason_code_counts": report["reason_code_counts"],
    }
    log_event("ranking_audit", "done", **result)
    return result


def load_audit_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_audit_config(config: dict[str, Any]) -> None:
    execution = config.get("execution", {})
    artifacts = config.get("artifacts", {})
    if execution.get("provider") != "modal":
        raise ValueError("Ranking audit must run on Modal.")
    if not str(artifacts.get("ranking_dir", "")).startswith("/artifacts"):
        raise ValueError("Audit ranking_dir must be mounted under /artifacts.")
    if not str(artifacts.get("audit_dir", artifacts.get("ranking_dir", ""))).startswith("/artifacts"):
        raise ValueError("Audit output must be mounted under /artifacts.")


def _submission_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    ranks = [int(float(row.get("rank", 0))) for row in rows]
    scores = _float_column(rows, "score")
    worker_ids = _string_column(rows, "worker_id")
    expected = list(range(1, len(ranks) + 1))
    return {
        "n_rows": len(rows),
        "rank_contiguous": ranks == expected,
        "score_nonincreasing": all(left >= right for left, right in zip(scores, scores[1:])),
        "duplicate_worker_count": len(worker_ids) - len(set(worker_ids)),
        "score": _quality_summary(scores),
    }


def _rank_slice_summary(rows: list[dict[str, str]], *, low_quality_threshold: float) -> dict[str, Any]:
    quality = _float_column(rows, "quality_score")
    reasons = Counter(_string_column(rows, "reason_code"))
    cluster_ids = _string_column(rows, "new_cluster_id")
    cluster_summary = _cluster_distribution_summary(cluster_ids)
    parent_cluster_ids = _parent_cluster_ids(rows)
    parent_cluster_summary = _cluster_distribution_summary(parent_cluster_ids)
    split_applied = [_truthy(row.get("large_cluster_split_applied", "")) for row in rows]
    return {
        "n_rows": len(rows),
        "mean_quality": _mean(quality),
        "min_quality": min(quality) if quality else 0.0,
        "low_quality_count": sum(value < low_quality_threshold for value in quality),
        "low_quality_fraction": _mean([1.0 if value < low_quality_threshold else 0.0 for value in quality]),
        "unique_cluster_count": len(set(cluster_ids)),
        "unique_clusters": len(set(cluster_ids)),
        "cluster_coverage_fraction": len(set(cluster_ids)) / len(rows) if rows and cluster_ids else 0.0,
        "dominant_cluster_id": cluster_summary["dominant_cluster_id"],
        "dominant_cluster_count": cluster_summary["dominant_cluster_count"],
        "dominant_cluster_fraction": cluster_summary["dominant_cluster_fraction"],
        "largest_cluster_id": cluster_summary["dominant_cluster_id"],
        "largest_cluster_count": cluster_summary["dominant_cluster_count"],
        "largest_cluster_fraction": cluster_summary["dominant_cluster_fraction"],
        "top_cluster_count": cluster_summary["dominant_cluster_count"],
        "cluster_gini": cluster_summary["cluster_gini"],
        "mean_per_cluster_count": cluster_summary["mean_per_cluster_count"],
        "top_cluster_counts": cluster_summary["top_cluster_counts"],
        "large_cluster_split_count": int(sum(split_applied)),
        "large_cluster_split_fraction": _mean([1.0 if value else 0.0 for value in split_applied]),
        "unique_parent_cluster_count": len(set(parent_cluster_ids)),
        "unique_parent_clusters": len(set(parent_cluster_ids)),
        "parent_largest_cluster_id": parent_cluster_summary["dominant_cluster_id"],
        "parent_largest_cluster_count": parent_cluster_summary["dominant_cluster_count"],
        "parent_largest_cluster_fraction": parent_cluster_summary["dominant_cluster_fraction"],
        "parent_cluster_gini": parent_cluster_summary["cluster_gini"],
        "parent_top_cluster_counts": parent_cluster_summary["top_cluster_counts"],
        "mean_grammar_score": _feature_mean(rows, ["grammar_score_component", "grammar_score"]),
        "mean_old_support_novelty": _feature_mean(rows, ["old_novelty_score", "old_knn_distance"]),
        "mean_new_batch_support": _feature_mean(rows, ["new_density_score", "new_batch_density"]),
        "mean_old_knn_distance": _feature_mean(rows, ["old_knn_distance"]),
        "mean_new_batch_density": _feature_mean(rows, ["new_batch_density"]),
        "corruption_negative_count": _bool_count(rows, "is_corruption"),
        "corruption_negative_fraction": _mean([1.0 if _truthy(row.get("is_corruption", "")) else 0.0 for row in rows]),
        "reason_code_counts": dict(reasons),
    }


def _candidate_eval_summary(
    candidates: list[dict[str, str]],
    *,
    top_ks: Iterable[int],
    low_quality_threshold: float,
) -> dict[str, Any]:
    ranked = sorted(candidates, key=lambda row: int(float(row.get("rank", 0))))
    summary = {
        "n_rows": len(ranked),
        "positive_rate": _mean([float(row.get("label", 0.0)) for row in ranked]),
        "top_k": {},
    }
    for k in top_ks:
        subset = ranked[: int(k)]
        labels = [int(float(row.get("label", 0))) for row in subset]
        quality = _float_column(subset, "quality_score")
        cluster_ids = _string_column(subset, "new_cluster_id")
        cluster_summary = _cluster_distribution_summary(cluster_ids)
        parent_cluster_ids = _parent_cluster_ids(subset)
        parent_cluster_summary = _cluster_distribution_summary(parent_cluster_ids)
        split_applied = [_truthy(row.get("large_cluster_split_applied", "")) for row in subset]
        summary["top_k"][str(k)] = {
            "positive_count": sum(labels),
            "positive_fraction": _mean(labels),
            "low_quality_count": sum(value < low_quality_threshold for value in quality),
            "unique_cluster_count": len(set(cluster_ids)),
            "unique_clusters": len(set(cluster_ids)),
            "cluster_coverage_fraction": len(set(cluster_ids)) / len(subset) if subset and cluster_ids else 0.0,
            "dominant_cluster_id": cluster_summary["dominant_cluster_id"],
            "dominant_cluster_count": cluster_summary["dominant_cluster_count"],
            "dominant_cluster_fraction": cluster_summary["dominant_cluster_fraction"],
            "largest_cluster_id": cluster_summary["dominant_cluster_id"],
            "largest_cluster_count": cluster_summary["dominant_cluster_count"],
            "largest_cluster_fraction": cluster_summary["dominant_cluster_fraction"],
            "top_cluster_count": cluster_summary["dominant_cluster_count"],
            "cluster_gini": cluster_summary["cluster_gini"],
            "mean_per_cluster_count": cluster_summary["mean_per_cluster_count"],
            "top_cluster_counts": cluster_summary["top_cluster_counts"],
            "large_cluster_split_count": int(sum(split_applied)),
            "large_cluster_split_fraction": _mean([1.0 if value else 0.0 for value in split_applied]),
            "unique_parent_cluster_count": len(set(parent_cluster_ids)),
            "unique_parent_clusters": len(set(parent_cluster_ids)),
            "parent_largest_cluster_id": parent_cluster_summary["dominant_cluster_id"],
            "parent_largest_cluster_count": parent_cluster_summary["dominant_cluster_count"],
            "parent_largest_cluster_fraction": parent_cluster_summary["dominant_cluster_fraction"],
            "parent_cluster_gini": parent_cluster_summary["cluster_gini"],
            "parent_top_cluster_counts": parent_cluster_summary["top_cluster_counts"],
            "mean_grammar_score": _feature_mean(subset, ["grammar_score_component", "grammar_score"]),
            "mean_old_support_novelty": _feature_mean(subset, ["old_novelty_score", "old_knn_distance"]),
            "mean_new_batch_support": _feature_mean(subset, ["new_density_score", "new_batch_density"]),
            "corruption_negative_count": _bool_count(subset, "is_corruption"),
            "corruption_negative_fraction": _mean([1.0 if _truthy(row.get("is_corruption", "")) else 0.0 for row in subset]),
            "reason_code_counts": dict(Counter(_string_column(subset, "reason_code"))),
        }
    return summary


def _grammar_diagnostics_summary(
    rows: list[dict[str, str]],
    *,
    low_quality_threshold: float,
    n_examples: int,
) -> dict[str, Any] | None:
    if not rows or "token_nll_p95" not in rows[0]:
        return None
    present_rows = [row for row in rows if _truthy(row.get("grammar_feature_present", "true"))]
    if not present_rows:
        return {
            "n_rows": len(rows),
            "present_count": 0,
            "present_fraction": 0.0,
            "token_nll_p95": _quality_summary([]),
            "rare_phrase_fraction": _quality_summary([]),
            "transition_nll_p95": _quality_summary([]),
            "longest_unseen_phrase_len": _quality_summary([]),
            "quality_correlation_token_nll_p95": 0.0,
            "old_novelty_correlation_token_nll_p95": 0.0,
            "new_density_correlation_token_nll_p95": 0.0,
            "top_surprisal_examples": [],
            "top_surprisal_low_quality_count": 0,
            "top_surprisal_mean_quality": 0.0,
            "top_surprisal_reason_code_counts": {},
        }

    top_examples = sorted(present_rows, key=lambda row: _to_float(row.get("token_nll_p95", "0")), reverse=True)[:n_examples]
    top_quality = _float_column(top_examples, "quality_score")
    token_nll = _float_column(present_rows, "token_nll_p95")
    quality = _float_column(present_rows, "quality_score")
    old_novelty = _float_column(present_rows, "old_novelty_score")
    new_density = _float_column(present_rows, "new_density_score")
    return {
        "n_rows": len(rows),
        "present_count": len(present_rows),
        "present_fraction": len(present_rows) / len(rows) if rows else 0.0,
        "token_nll_p95": _quality_summary(token_nll),
        "rare_phrase_fraction": _quality_summary(_float_column(present_rows, "rare_phrase_fraction")),
        "transition_nll_p95": _quality_summary(_float_column(present_rows, "transition_nll_p95")),
        "longest_unseen_phrase_len": _quality_summary(_float_column(present_rows, "longest_unseen_phrase_len")),
        "quality_correlation_token_nll_p95": _correlation(token_nll, quality),
        "old_novelty_correlation_token_nll_p95": _correlation(token_nll, old_novelty),
        "new_density_correlation_token_nll_p95": _correlation(token_nll, new_density),
        "top_surprisal_examples": top_examples,
        "top_surprisal_low_quality_count": sum(value < low_quality_threshold for value in top_quality),
        "top_surprisal_mean_quality": _mean(top_quality),
        "top_surprisal_reason_code_counts": dict(Counter(_string_column(top_examples, "reason_code"))),
    }


def _low_quality_examples(
    rows: list[dict[str, str]],
    *,
    low_quality_threshold: float,
    n_examples: int,
) -> list[dict[str, str]]:
    low_rows = [row for row in rows if _to_float(row.get("quality_score", "1.0")) < low_quality_threshold]
    return sorted(low_rows, key=lambda row: _to_float(row.get("quality_score", "1.0")))[:n_examples]


def _cluster_distribution_summary(clusters: list[str]) -> dict[str, Any]:
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


def _parent_cluster_ids(rows: list[dict[str, str]]) -> list[str]:
    parent_ids: list[str] = []
    for row in rows:
        parent = str(row.get("new_cluster_parent_id", ""))
        cluster = str(row.get("new_cluster_id", ""))
        parent_ids.append(parent or cluster)
    return [cluster_id for cluster_id in parent_ids if cluster_id]


def _quality_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "count": float(len(sorted_values)),
        "mean": _mean(sorted_values),
        "min": float(sorted_values[0]),
        "p10": _percentile(sorted_values, 10),
        "p50": _percentile(sorted_values, 50),
        "p90": _percentile(sorted_values, 90),
        "max": float(sorted_values[-1]),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _float_column(rows: list[dict[str, str]], key: str) -> list[float]:
    return [_to_float(row[key]) for row in rows if key in row and row[key] != ""]


def _feature_mean(rows: list[dict[str, str]], keys: list[str]) -> float:
    for key in keys:
        values = _float_column(rows, key)
        if values:
            return _mean(values)
    return 0.0


def _bool_count(rows: list[dict[str, str]], key: str) -> int:
    return int(sum(1 for row in rows if _truthy(row.get(key, ""))))


def _string_column(rows: list[dict[str, str]], key: str) -> list[str]:
    return [str(row.get(key, "")) for row in rows if str(row.get(key, ""))]


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return float(sum(numbers) / len(numbers)) if numbers else 0.0


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


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = sum(value * value for value in left_centered) ** 0.5
    right_norm = sum(value * value for value in right_centered) ** 0.5
    denom = left_norm * right_norm
    return float(numerator / denom) if denom > 1.0e-12 else 0.0


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
