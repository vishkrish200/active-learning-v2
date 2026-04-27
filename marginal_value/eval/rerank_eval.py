from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from marginal_value.eval.ablation_eval import average_precision_at_k, cluster_diversity_at_k, ndcg_at_k, precision_at_k
from marginal_value.logging_utils import log_event


DEFAULT_K_VALUES = [10, 50, 100, 200]
DEFAULT_LOW_QUALITY_THRESHOLD = 0.45


def run_rerank_eval(
    config: dict[str, Any],
    *,
    candidate_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    allow_local_execution: bool = False,
) -> dict[str, Any]:
    validate_rerank_eval_config(config, allow_local_execution=allow_local_execution)
    artifacts = config["artifacts"]
    candidates = Path(candidate_path) if candidate_path is not None else Path(artifacts["candidate_path"])
    output = Path(output_dir) if output_dir is not None else Path(artifacts["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    log_event("rerank_eval", "start", candidate_path=str(candidates), output_dir=str(output))
    rows = _read_csv(candidates)
    log_event("rerank_eval", "candidate_read_done", n_rows=len(rows))

    eval_config = config.get("eval", {})
    k_values = [int(value) for value in eval_config.get("k_values", DEFAULT_K_VALUES)]
    report = evaluate_rerank_variants(
        rows,
        k_values=k_values,
        low_quality_threshold=float(eval_config.get("low_quality_threshold", DEFAULT_LOW_QUALITY_THRESHOLD)),
    )
    report["mode"] = "full"
    report["candidate_path"] = str(candidates)

    report_path = output / "rerank_eval_report_full.json"
    summary_path = output / "rerank_eval_summary_full.csv"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_rows(summary_path, report["variants"])

    result = {
        "n_rows": len(rows),
        "variants": sorted(report["variants"]),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
    }
    primary_key = "best_by_cluster_repeat_at_100"
    if primary_key in report:
        result[primary_key] = report[primary_key]
    log_event("rerank_eval", "done", **result)
    return result


def evaluate_rerank_variants(
    rows: Sequence[dict[str, object]],
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    low_quality_threshold: float = DEFAULT_LOW_QUALITY_THRESHOLD,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty.")
    ks = [int(value) for value in k_values]
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("k_values must contain positive integers.")
    if not 0.0 <= low_quality_threshold <= 1.0:
        raise ValueError("low_quality_threshold must be in [0, 1].")

    raw_order = _sort_by_float_column(rows, "final_score", descending=True)
    raw_top_score_means = _top_score_means(rows, raw_order, ks)
    variants: dict[str, dict[str, Any]] = {
        "raw_final_score": _summarize_order(
            rows,
            raw_order,
            k_values=ks,
            low_quality_threshold=low_quality_threshold,
            raw_top_score_means=raw_top_score_means,
        )
    }

    existing_order = _existing_rerank_order(rows)
    if existing_order is not None:
        variants["existing_rerank"] = _summarize_order(
            rows,
            existing_order,
            k_values=ks,
            low_quality_threshold=low_quality_threshold,
            raw_top_score_means=raw_top_score_means,
        )

    report: dict[str, Any] = {
        "n_rows": len(rows),
        "k_values": ks,
        "low_quality_threshold": float(low_quality_threshold),
        "variants": variants,
    }
    for k in ks:
        report[f"best_by_cluster_repeat_at_{k}"] = _best_by_cluster_repeat(variants, k)
    return report


def validate_rerank_eval_config(config: dict[str, Any], *, allow_local_execution: bool = False) -> None:
    execution = _required_mapping(config, "execution")
    artifacts = _required_mapping(config, "artifacts")
    if execution.get("provider") != "modal" and not allow_local_execution:
        raise ValueError("Rerank evaluation must be dispatched through Modal.")
    candidate_path = Path(str(artifacts.get("candidate_path", "")))
    output_dir = Path(str(artifacts.get("output_dir", "")))
    if not candidate_path.is_absolute():
        raise ValueError("artifacts.candidate_path must be absolute.")
    if not output_dir.is_absolute():
        raise ValueError("artifacts.output_dir must be absolute.")

    eval_config = config.get("eval", {})
    k_values = eval_config.get("k_values", DEFAULT_K_VALUES)
    if not k_values or any(int(value) <= 0 for value in k_values):
        raise ValueError("eval.k_values must contain positive integers.")
    low_quality_threshold = float(eval_config.get("low_quality_threshold", DEFAULT_LOW_QUALITY_THRESHOLD))
    if not 0.0 <= low_quality_threshold <= 1.0:
        raise ValueError("eval.low_quality_threshold must be in [0, 1].")


def load_rerank_eval_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize_order(
    rows: Sequence[dict[str, object]],
    order: Sequence[int],
    *,
    k_values: Sequence[int],
    low_quality_threshold: float,
    raw_top_score_means: dict[int, float],
) -> dict[str, Any]:
    ordered_rows = [rows[idx] for idx in order]
    labels = [_to_int(row.get("label", 0)) for row in ordered_rows]
    clusters = [_cluster_id(row, fallback=idx) for idx, row in zip(order, ordered_rows)]
    qualities = [_to_float(row.get("quality_score", 1.0), default=1.0) for row in ordered_rows]
    final_scores = [_to_float(row.get("final_score", 0.0), default=0.0) for row in ordered_rows]
    total_clusters = max(1, len(set(clusters)))

    metrics: dict[str, float] = {
        "score_mean": float(np.mean(final_scores)) if final_scores else 0.0,
        "positive_rate": float(np.mean(np.asarray(labels, dtype=int) > 0)) if labels else 0.0,
    }
    top_k_reason_counts: dict[str, dict[str, int]] = {}
    for k in k_values:
        k_eff = min(int(k), len(ordered_rows))
        top_labels = labels[:k_eff]
        top_clusters = clusters[:k_eff]
        top_qualities = qualities[:k_eff]
        top_scores = final_scores[:k_eff]
        reason_counts = Counter(str(row.get("reason_code", "")) or "UNKNOWN" for row in ordered_rows[:k_eff])
        unique_cluster_count = len(set(top_clusters))
        raw_score_mean = raw_top_score_means.get(int(k), 0.0)
        top_score_mean = float(np.mean(top_scores)) if top_scores else 0.0

        metrics[f"precision@{k}"] = precision_at_k(labels, k_eff)
        metrics[f"ap@{k}"] = average_precision_at_k(labels, k_eff)
        metrics[f"ndcg@{k}"] = ndcg_at_k(labels, k_eff)
        metrics[f"cluster_diversity@{k}"] = cluster_diversity_at_k(labels, clusters, k_eff)
        metrics[f"unique_cluster_count@{k}"] = float(unique_cluster_count)
        metrics[f"cluster_repeat_count@{k}"] = float(k_eff - unique_cluster_count)
        metrics[f"cluster_coverage_fraction@{k}"] = float(unique_cluster_count / total_clusters)
        metrics[f"mean_quality@{k}"] = float(np.mean(top_qualities)) if top_qualities else 0.0
        metrics[f"low_quality_count@{k}"] = float(sum(quality < low_quality_threshold for quality in top_qualities))
        metrics[f"low_quality_rate@{k}"] = metrics[f"low_quality_count@{k}"] / k_eff if k_eff else 0.0
        metrics[f"mean_final_score@{k}"] = top_score_mean
        metrics[f"score_retention_vs_raw@{k}"] = top_score_mean / raw_score_mean if raw_score_mean > 0 else 0.0
        top_k_reason_counts[str(k)] = {reason: int(count) for reason, count in sorted(reason_counts.items())}

    return {
        "order_source": "candidate_rows",
        "metrics": metrics,
        "top_k_reason_counts": top_k_reason_counts,
    }


def _existing_rerank_order(rows: Sequence[dict[str, object]]) -> list[int] | None:
    ranks = [_to_float(row.get("rank", np.nan), default=np.nan) for row in rows]
    if np.all(np.isfinite(ranks)):
        return sorted(range(len(rows)), key=lambda idx: (ranks[idx], idx))
    rerank_scores = [_to_float(row.get("rerank_score", np.nan), default=np.nan) for row in rows]
    if np.all(np.isfinite(rerank_scores)):
        return sorted(range(len(rows)), key=lambda idx: (-rerank_scores[idx], idx))
    return None


def _sort_by_float_column(rows: Sequence[dict[str, object]], column: str, *, descending: bool) -> list[int]:
    values = [_to_float(row.get(column, np.nan), default=np.nan) for row in rows]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Column '{column}' must contain finite numeric values.")
    if descending:
        return sorted(range(len(rows)), key=lambda idx: (-values[idx], idx))
    return sorted(range(len(rows)), key=lambda idx: (values[idx], idx))


def _top_score_means(rows: Sequence[dict[str, object]], order: Sequence[int], k_values: Sequence[int]) -> dict[int, float]:
    scores = [_to_float(rows[idx].get("final_score", 0.0), default=0.0) for idx in order]
    return {int(k): float(np.mean(scores[: min(int(k), len(scores))])) if scores else 0.0 for k in k_values}


def _best_by_cluster_repeat(variants: dict[str, dict[str, Any]], k: int) -> str:
    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float, str]:
        name, variant = item
        metrics = variant.get("metrics", {})
        return (
            float(metrics.get(f"cluster_repeat_count@{k}", float("inf"))),
            -float(metrics.get(f"unique_cluster_count@{k}", 0.0)),
            -float(metrics.get(f"ndcg@{k}", 0.0)),
            -float(metrics.get(f"mean_final_score@{k}", 0.0)),
            name,
        )

    return min(variants.items(), key=sort_key)[0]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_summary_rows(path: Path, variants: dict[str, dict[str, Any]]) -> None:
    rows = []
    for name, variant in sorted(variants.items()):
        row: dict[str, object] = {"variant": name}
        row.update(variant.get("metrics", {}))
        rows.append(row)
    _write_rows(path, rows)


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cluster_id(row: dict[str, object], *, fallback: int) -> str:
    value = row.get("new_cluster_id", "")
    text = str(value).strip()
    return text if text else str(fallback)


def _to_float(value: object, *, default: float) -> float:
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
        raise ValueError(f"Rerank eval config must include a '{key}' object.")
    return value
